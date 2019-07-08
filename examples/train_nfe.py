import numpy as np
import tensorflow as tf

import sys, os, time, json, gzip
import argparse
from itertools import izip

sys.path.insert(0, "../NEMO/")
import nfe
import data
from coordinates import batch_to_pdb, batch_trajectory_to_pdb

# Learning parameters
parser = argparse.ArgumentParser(description="Train a Neural Folding Engine.")
parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs")
parser.add_argument("--enrich", action="store_true", help="Enrich with sequence database")
parser.add_argument("--profiles", action="store_true", help="Train with profiles")
parser.add_argument("--restore", type=str, default=None, help="Restore from checkpoint")
parser.add_argument("--length", type=int, default=200, help="Maximum length")
ARGS = parser.parse_args()

ANNEAL_STEPS = 300000
WARMUP = 10000
batch_size = 10
batch_size_test = 16
frequencies = {
    "valid": 50,
    "dense": 50,
    "trajectory": 50,
    "pdb": 50,
    "save": 100
}

# Build the model
print "Compiling model"
start = time.time()
NFE = nfe.NeuralFoldingEngine(
    num_gpus=ARGS.num_gpus, use_profiles=ARGS.profiles
)
elapsed = time.time() - start
print " compiles in " + str(elapsed) + " s\n"

# Load data
CATH_FILE = "data/cath_L200_class123.txt"
CATH_SETS_FILE = "data/cath_L200_class123_sets.json"
dataset = data.DataSet(
    cath_file = CATH_FILE,
    cath_sets_file = CATH_SETS_FILE,
    max_length = ARGS.length,
    num_gpus = ARGS.num_gpus,
    placeholder_sets = NFE.placeholder_sets
)

# Load auxilliary sequence data
prefix = "data/"
if ARGS.enrich:
    seq_file = prefix + "sequences_cath_L200.txt"
    weights_file = prefix + "sequence_weights_cath_L200.txt"
    dataset.load_alignments(seq_file, weights_file)
if ARGS.profiles:
    profile_file = prefix + "profiles_cath_L200.json"
    profile_dict = dataset.load_profiles(profile_file)

# Build experiment paths
base_folder = time.strftime("log/%y%b%d_%I%M%p/", time.localtime())
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
folders = ["train_full", "train_trajectory", "valid_full", "valid_trajectory"]
paths = {}
for folder in folders:
    path = base_folder + "/" + folder + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    paths[folder] = path

# Data output
output_keys = [
    "score", "coordinates_fine", "trajectory", "coarse_target", "logp_distances", 
    "contact_precision", "contact_recall"
]
def _output_sets(out_file, out_sets, batch_ids):
    """ Output """
    print_keys = ["contact_precision", "contact_recall", "logp_distances"]
    with open(out_file, "w") as f:
        f.write("cath_id\tgpu\t" + "\t".join(print_keys) + "\n")
        for gpu_ix in xrange(ARGS.num_gpus):
            concat_vals = [out_sets[gpu_ix][print_key] for print_key in print_keys]
            concat_stack = np.stack(concat_vals, axis=1)
            for iy in xrange(len(batch_ids[gpu_ix])):
                out_str = map(str,concat_stack[iy].tolist())
                prefix = batch_ids[gpu_ix][iy] + "\t" + str(gpu_ix) + "." + str(iy) + "\t"
                f.write(prefix + "\t".join(out_str) + "\n")

# Write hyperparamters and dimensions as JSON
with open(base_folder + "hyperparams.json", "w") as f:
    json.dump(NFE.hyperparams, f, indent=4)

# Launch session
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=False
)
with tf.Session(config=config) as sess:
    # Initialization
    print "Initializing variables"
    start = time.time()
    init = tf.global_variables_initializer()
    sess.run(init)
    elapsed = time.time() - start
    print " initializes in " + str(elapsed) + " s\n"

    # Build batch summaries and TensorBoard writer
    summaries = NFE.summaries
    print "Setting up saver"
    start = time.time()
    summary_writer = tf.summary.FileWriter(base_folder, sess.graph)

    # Variable saving
    saver = tf.train.Saver()
    if ARGS.restore is not None:
        saver.restore(sess, ARGS.restore)
    elapsed = time.time() - start
    print " set up in " + str(elapsed) + " s\n"

    # Training loop
    start = time.time()
    for i in xrange(1000000):
        global_step = sess.run([NFE.global_step])[0]

        # Test
        score, grad_norm_gen = np.nan, np.nan

        # Build the batch
        feed_dict, batch_ids = dataset.sample_batch_train()

        # Also anneal the annealables
        for gpu_ix in xrange(ARGS.num_gpus):
            anneal_frac = np.minimum(1.0, float(global_step) / float(ANNEAL_STEPS))
            feed_dict[NFE.placeholder_sets[gpu_ix]["beta_anneal"]] = np.minimum(anneal_frac, 1.)
            feed_dict[NFE.placeholder_sets[gpu_ix]["langevin_steps"]] = 250
            feed_dict[NFE.placeholder_sets[gpu_ix]["native_init_prob"]] = 0.2
            feed_dict[NFE.placeholder_sets[gpu_ix]["training"]] = True
            feed_dict[NFE.placeholder_sets[gpu_ix]["native_unfold_randomize"]] = False
        print "Beta " + str(feed_dict[NFE.placeholder_sets[gpu_ix]["beta_anneal"]]) + " schedule " + str(feed_dict[NFE.placeholder_sets[gpu_ix]["langevin_steps"]])


        summary_op = summaries["gen_batch"]
        if global_step % frequencies["dense"] == 0:
            summary_op = summaries["gen_dense"]
        # Generic ops
        train_gen_ops = [summary_op, NFE.train_gen, NFE.grad_norm_gen]
        for gpu_ix in xrange(ARGS.num_gpus):
            train_gen_ops += [NFE.tensor_sets[gpu_ix][key] for key in output_keys]
        try:
            # Run the ops
            train_outs = sess.run(train_gen_ops, feed_dict=feed_dict)

            # Process output
            summary_gen, grad_norm_gen = train_outs[0], train_outs[2]
            offset_ix = 3
            train_out_sets = []
            for gpu_ix in xrange(ARGS.num_gpus):
                train_subset = train_outs[offset_ix:offset_ix + len(output_keys)]
                train_out_sets.append(dict(zip(output_keys, train_subset)))
                offset_ix += len(output_keys)
            score = train_out_sets[0]["score"]

            # Write summaries
            summary_writer.add_summary(summary_gen, global_step)
        except Exception as e:
            print e

        # Status report
        print global_step, score, time.time() - start, grad_norm_gen

        # Coordinate output
        if np.isfinite(score):
            if global_step % frequencies["pdb"] == 0:
                for gpu_ix in xrange(ARGS.num_gpus):
                    # Write pdb
                    pdb = batch_to_pdb(feed_dict[NFE.placeholder_sets[gpu_ix]["lengths"]],
                                       feed_dict[NFE.placeholder_sets[gpu_ix]["sequences"]],
                                       feed_dict[NFE.placeholder_sets[gpu_ix]["coordinates_target"]],
                                       train_out_sets[gpu_ix]["coordinates_fine"],
                                       feed_dict[NFE.placeholder_sets[gpu_ix]["structure_mask"]])
                    pdb_fn = "step" + "{0:07d}".format(global_step) + "_" + str(gpu_ix) + ".pdb.gz"
                    with gzip.open(paths["train_full"] + pdb_fn, "w") as f:
                        f.write("\n".join(pdb) + "\n")
                out_file = paths["train_full"] + "step" + "{0:07d}".format(global_step) + ".txt"
                _output_sets(out_file, train_out_sets, batch_ids)
            if global_step % frequencies["trajectory"] == 0:
                for gpu_ix in xrange(ARGS.num_gpus):
                    # Plot coarse folding trajectory
                    pdb = batch_trajectory_to_pdb(
                        feed_dict[NFE.placeholder_sets[gpu_ix]["lengths"]],
                        feed_dict[NFE.placeholder_sets[gpu_ix]["sequences"]],
                        train_out_sets[gpu_ix]["coarse_target"],
                        train_out_sets[gpu_ix]["trajectory"],
                        feed_dict[NFE.placeholder_sets[gpu_ix]["structure_mask"]])
                    pdb_fn = "step" + "{0:07d}".format(global_step) + "_" + str(gpu_ix) + ".pdb.gz"
                    with gzip.open(paths["train_trajectory"] + pdb_fn, "w") as f:
                        f.write("\n".join(pdb) + "\n")

        # Test set evaluation
        if global_step % frequencies["valid"] == 0 and global_step >= WARMUP:
            for key in ["valid_A", "valid_T", "valid_H"]:
                # Make test set minibatch
                feed_dict, batch_ids = dataset.sample_batch_test(key)
                # Anneal annealables
                for gpu_ix in xrange(ARGS.num_gpus):
                    feed_dict[NFE.placeholder_sets[gpu_ix]["langevin_steps"]] = 250
                print "Schedule " + str(feed_dict[NFE.placeholder_sets[gpu_ix]["langevin_steps"]])

                # Summaries
                summary_op = summaries[key]

                # Generic ops
                valid_ops = [summary_op]
                for gpu_ix in xrange(ARGS.num_gpus):
                    valid_ops += [NFE.tensor_sets[gpu_ix][output] for output in output_keys]
                summary = valid_out[0]
                summary_writer.add_summary(summary, global_step)

                # Test output
                offset_ix = 1
                valid_out_sets = []
                for gpu_ix in xrange(ARGS.num_gpus):
                    test_subset = valid_out[offset_ix:offset_ix + len(output_keys)]
                    valid_out_sets.append(dict(zip(output_keys, test_subset)))
                    offset_ix += len(output_keys)

                out_file = paths["valid_full"] + "step" + "{0:07d}".format(global_step) + key + ".txt"
                _output_sets(out_file, valid_out_sets, batch_ids)

                for gpu_ix in xrange(ARGS.num_gpus):
                    # Write pdb
                    pdb = batch_to_pdb(feed_dict[NFE.placeholder_sets[gpu_ix]["lengths"]],
                                       feed_dict[NFE.placeholder_sets[gpu_ix]["sequences"]],
                                       feed_dict[NFE.placeholder_sets[gpu_ix]["coordinates_target"]],
                                       valid_out_sets[gpu_ix]["coordinates_fine"],
                                       feed_dict[NFE.placeholder_sets[gpu_ix]["structure_mask"]])
                    pdb_fn = "step" + "{0:07d}".format(global_step) + key + "_" + str(gpu_ix) + ".pdb.gz"
                    with gzip.open(paths["valid_full"] + pdb_fn, "w") as f:
                        f.write("\n".join(pdb) + "\n")
                    # Plot coarse folding trajectory
                    pdb = batch_trajectory_to_pdb(
                        feed_dict[NFE.placeholder_sets[gpu_ix]["lengths"]],
                        feed_dict[NFE.placeholder_sets[gpu_ix]["sequences"]],
                        valid_out_sets[gpu_ix]["coarse_target"],
                        valid_out_sets[gpu_ix]["trajectory"],
                        feed_dict[NFE.placeholder_sets[gpu_ix]["structure_mask"]])
                    pdb_fn = "step" + \
                        "{0:07d}".format(global_step) + key + "_" + str(gpu_ix) + ".pdb.gz"
                    with gzip.open(paths["valid_trajectory"] + pdb_fn, "w") as f:
                        f.write("\n".join(pdb) + "\n")

        if global_step % frequencies["save"] == 0:
            saver.save(sess, base_folder + "model",
                       global_step=global_step)
    summary_writer.close()
