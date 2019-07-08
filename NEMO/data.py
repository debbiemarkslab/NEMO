import gzip
import json
import time
import random
import numpy as np
from collections import defaultdict
from itertools import izip

from scipy.spatial.distance import squareform

class DataSet:
    def __init__(self, max_length=100, cath_file=None, cath_sets_file=None,
        num_gpus=1, placeholder_sets=None, use_ss=False, use_ec=False):
        np.random.seed(42)    

        """ Load a folding dataset """
        self.test_keys = ["valid_A", "valid_T", "valid_H"]
        self.use_profiles = False
        self.use_enrich = False
        self.use_couplings = False
        self.use_ss = use_ss
        self.num_gpus = num_gpus
        self.placeholder_sets = placeholder_sets

        # Build alphabet map
        self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
        self.letter_to_index = {}
        for i,letter in enumerate(self.alphabet):
            self.letter_to_index[letter] = i

        # Load fold dataset
        self.length = max_length
        self.fold_dict, self.test_train, self.length = \
            self._load_fold_set(cath_file, cath_sets_file)
        self.fold_ids = self.fold_dict.keys()
        self.fold_data = self.fold_dict.values()

        # Build size-dependent batching
        self.subsets, self.subset_weights = self._build_size_subsets()

        # Feed dict keys
        self.feed_keys = [
            "sequences", "coordinates_target", "secondary_structure",
            "lengths", "structure_mask"
        ]
        return

    def _load_fold_set(self, domain_file, set_file=None):
        """ Load a set of domains """
        fold_dict = {}
        with open(domain_file) as f:
            print "Loading domain file into memory"
            start = time.time()
            lines = f.readlines()
            elapsed = time.time() - start
            print "Loads in " + str(elapsed)

            print "Parsing " + str(len(lines)) + " domains"
            start = time.time()
            for i, line in enumerate(lines):
                name, node, data = line.split("\t")
                obj = json.loads(data)
                # 
                fold_dict[name] = obj
                if (i + 1) % 5000 == 0:
                    print "\t" + str(i + 1) + " domains loaded"
            elapsed = time.time() - start
        print " parses in " + str(elapsed) + " sec\n"

        # Load test and training sets
        test_train = None
        if set_file is not None:
            with open(set_file) as f:
                test_train = json.load(f)

        # Find the maximum length in loaded set
        lengths = [entry["length"] for entry in fold_dict.values()]
        max_length = min([max(lengths), self.length])

        # Purge test set members above MAX_LENGTH by reweighting their probability
        for key in self.test_keys:  
            ids, weights = test_train[key]
            lengths = [fold_dict[name]["length"] for name in ids]
            weights = [
                w if lengths[ix] <= max_length else 0 
                for ix, w in enumerate(weights)
            ]
            total = sum(weights)
            test_train[key][1] = [
                w / total for w in weights
            ]
        return fold_dict, test_train, max_length

    def load_profiles(self, filename):
        """ Load a set of sequence profiles from a json flatfile """
        profile_dict = {}
        i = 0
        start = time.time()
        with open(filename, "r") as f:
            for line in f:
                cath_id, json_entry = line.split("\t")
                if cath_id in self.fold_ids:
                    profile = np.asarray(
                        json.loads(json_entry), dtype=np.float32
                    )
                    profile_dict[cath_id] = profile
                    i += 1
                    if i % 5000 is 0:
                        print "Loaded {} entries, {} s".format(i, time.time() - start)

        # Concatenate profiles to batch features
        self.profiles = profile_dict
        self.use_profiles = True
        return

    def load_alignments(self, seq_file, weights_file):
        """ Load sequence alignments for each protein """
        print "Loading alignments"
        start = time.time()
        seq_dict = {}
        weight_dict = {}
        i, total = 0, 0
        with open(seq_file) as f:
            for line in f:
                cath_id, seqs = line.split("\t")
                if cath_id in self.fold_ids:
                    seqs = [seq.strip().split(":")[1] for seq in seqs.split(",")]
                    seq_dict[cath_id] = seqs
                    i += 1
                    total += len(seqs)
                    if i % 5000 == 0:
                        print "Loaded {} domains, {} sequences, {} s".format(i, total, time.time() - start)
        print "Loading weights"
        i, total = 0, 0
        with open(weights_file) as f:
            for line in f:
                cath_id, weights = line.split("\t")
                if cath_id in self.fold_ids:
                    weights = [float(weight.strip().split(":")[1]) for weight in weights.split(",")]
                    weight_dict[cath_id] = weights
                    # Compute all vs all weights
                    i += 1
                    total += len(weights)
                    if i % 5000 == 0:
                        print "Loaded {} domains, {} weights, {} s".format(i, total, time.time() - start)
        

        # Replace potentially absent keys
        missing_ids = list(set(self.fold_ids) - set(seq_dict.keys()))
        if len(missing_ids) > 0:
            missing_seqs = [self.fold_dict[cath_id]["seq"] for cath_id in missing_ids]
            missing_weights = [1.0 for cath_id in missing_ids]
            weight_dict.update(dict(zip(missing_ids,missing_weights)))
            seq_dict.update(dict(zip(missing_ids,missing_seqs)))
            print "Replaced missing keys:" + str(missing_ids)

        # Concatenate profiles to batch features
        self.align_dict = seq_dict
        self.weight_dict = weight_dict
        self.use_enrich = True
        return

    def load_couplings(self, couplings_file):
        """ Load covariation scores for each protein """
        coupling_dict = {}
        ix = 1
        print "Loading couplings"
        with open(couplings_file, "rb") as f:
            lines = f.readlines()
            for line in lines:
                cath_id, ec_json = line.split("\t")
                length, ec = json.loads(ec_json)
                ec_ix = None
                ec_val = None
                if len(ec) > 0:
                    ec_ix, ec_val = zip(*ec)
                    ec_ix = np.asarray(ec_ix, dtype=np.int32)
                    ec_val = np.asarray(ec_val, dtype=np.float32)
                coupling_dict[cath_id] = (length, ec_ix, ec_val)
                ix += 1
                if ix % 5000 == 0:
                    print str(ix) + " couplings loaded"
        self.couplings = coupling_dict
        self.use_couplings = True
        self.feed_keys += ["couplings"]
        return

    def _build_size_subsets(self):
        """ Build size sub-batching """

        coeffs = [8.631E-6, 7.2619E-5, 0.006]
        subset_lengths = range(20, self.length + 20, 20)
        subset_counts = [
            int(round(0.9 * 1./(coeffs[0]*x*x + coeffs[1]*x + coeffs[0])))
            for x in subset_lengths
        ]
        ids, weights = self.test_train["train"]

        id_lengths = [self.fold_dict[name]["length"] for name in ids]
        # Build subsets
        subsets = [{} for _ in subset_lengths]
        subset_weights = [0 for _ in subset_lengths]
        LB = 0
        for i, UB in enumerate(subset_lengths):
            if UB <= self.length:
                indices = [ix for ix, l in enumerate(
                    id_lengths) if l > LB and l <= UB]
                subsets[i]["ids"] = [ids[ix] for ix in indices]
                subsets[i]["weights"] = [weights[ix] for ix in indices]
                total = sum(subsets[i]["weights"])
                subset_weights[i] = total
                subsets[i]["weights"] = [w / total for w in subsets[i]["weights"]]
                subsets[i]["counts"] = subset_counts[i]
                LB = UB
        Z = sum(subset_weights)
        subset_weights = [float(s) / float(Z) for s in subset_weights]
        return subsets, subset_weights


    def _seq_to_onehot(self, seq, length):
        one_hot_seq = np.zeros((length, len(self.alphabet)), dtype=np.float32)
        for i,letter in enumerate(seq):
            if letter == "U":
                letter = "C"
            if letter in self.alphabet:
                j = self.letter_to_index[letter]
                one_hot_seq[i,j] = 1.
        return one_hot_seq


    def _ss_to_onehot(self, ss, length):
        one_hot_ss = np.zeros((length, 8), dtype=np.float32)
        for i, state in enumerate(ss):
            if state >= 0:
                one_hot_ss[i, state] = 1.0
        return one_hot_ss

    def _ss_to_onehot_3(self, ss, length):
        """
           0/0   pi helix
           1   bend
           2/0   alpha helix
           3   extended
           4/0   3-10 helix
           5/1   bridge
           6   turn
           7   coil
        """
        one_hot_ss = np.zeros((length, 3), dtype=np.float32)
        reduced_SS = [0,2,0,1,0,1,2,2]
        for i, state in enumerate(ss):
            if state >= 0:
                one_hot_ss[i, reduced_SS[state]] = 1.0
        return one_hot_ss

    def _build_minibatch(self, indices):
        """ Build a minibatch dictionary given a set of ids """
        minibatch_dict = {
            "sequences": [],
            "coordinates_target": [],
            "secondary_structure": [],
            "lengths": [],
            "structure_mask": []
        }
        if self.use_couplings:
            minibatch_dict["couplings"] = []

        # Find max length
        pad_length = max([self.fold_data[idx]["length"] for idx in indices])
        print str(len(indices)) + " domains of length " + str(pad_length)

        for idx in indices:
            protein = self.fold_data[idx]
            one_hot_seq = self._seq_to_onehot(protein["seq"], pad_length)
            one_hot_ss = self._ss_to_onehot(protein["SS"], pad_length)
            seq_length = protein["length"]

            # Profile concatenation
            if self.use_profiles and not self.use_ss:
                profile = self.profiles[self.fold_ids[idx]]
                profile_pad = np.concatenate(
                    (profile, np.zeros((pad_length - seq_length, 20))),
                    axis=0
                )
                one_hot_seq = np.concatenate(
                    (one_hot_seq, profile_pad), axis=1
                )
                one_hot_seq = one_hot_seq.astype(np.float32)
            elif self.use_ss:
                one_hot_ss_3 = self._ss_to_onehot_3(
                    protein["SS"], pad_length
                )
                one_hot_seq = np.concatenate(
                    (one_hot_seq, one_hot_ss_3), axis=1
                )

            # Reshape and pad the coordinate sets
            coord_types = [
                "N_coords",
                "CA_coords",
                "C_coords",
                "O_coords",
                "SCOM_coords"
            ]
            coord_sets = []
            for coord_type in coord_types:
                coord_sets.append(
                    np.asarray(protein[coord_type]).reshape((3, -1))
                )
            target_coords = np.concatenate(coord_sets).T
            target_coords_padded = np.concatenate(
                [target_coords, np.zeros((pad_length - seq_length, 15))]
            )

            # Structure mask for missing regions of crystal structures
            mask = np.concatenate(
                [np.asarray(protein["mask"]), 
                 np.zeros((pad_length - seq_length))
                ]
            )

            # Covariation scores
            if self.use_couplings:
                ij_length, ec_ix, ec_vals = self.couplings[self.fold_ids[idx]]
                ec_ij = np.zeros((ij_length))
                if ec_ix is not None:
                    ec_ij[ec_ix] = ec_vals
                delta = pad_length - seq_length
                ec = np.pad(
                    squareform(ec_ij), ((0,delta),(0,delta)), "constant"
                )
                minibatch_dict["couplings"].append(ec)

            # Then add these to the minibatch_list
            minibatch_dict["sequences"].append(one_hot_seq)
            minibatch_dict["coordinates_target"].append(target_coords_padded)
            minibatch_dict["secondary_structure"].append(one_hot_ss)
            minibatch_dict["lengths"].append(seq_length)
            minibatch_dict["structure_mask"].append(mask)

        # Flatten them into numpy arrays
        minibatch_dict = {
            key: np.asarray(value) for key, value in minibatch_dict.iteritems()
        }

        # Dimension types
        minibatch_dict["structure_mask"] = \
                np.array(minibatch_dict["structure_mask"], dtype=np.float32)

        # Remove NaNs 
        for key in minibatch_dict:
            minibatch_dict[key][np.isnan(minibatch_dict[key])] = 0
        return minibatch_dict


    def sample_batch_train(self):
        feed_dict = {}
        batch_id_sets = []
        for i in xrange(self.num_gpus):
            # Sample a size class
            ix = np.random.choice(range(len(self.subsets)), p=self.subset_weights)
            batch_ids = np.random.choice(
                self.subsets[ix]["ids"], self.subsets[ix]["counts"], p=self.subsets[ix]["weights"]
            ).tolist()
            batch_ix = [self.fold_ids.index(name) for name in batch_ids]
            batch_dict = self._build_minibatch(batch_ix)
            batch_id_sets.append(batch_ids)
            
            if self.use_enrich:
                SEQ_BETA = 5.
                # Swap in enriched sequences
                batch_seqs = []
                length = batch_dict["sequences"].shape[1]
                for ix, batch_id in enumerate(batch_ids):
                    weights = np.array(self.weight_dict[batch_id])
                    weights = np.exp(SEQ_BETA * weights)
                    weights = weights / np.sum(weights)
                    native_seq = self.fold_dict[batch_id]["seq"]
                    seq = np.random.choice(self.align_dict[batch_id], p=weights)
                    seq = "".join([
                        s if s is not "-" else native_seq[seq_ix] 
                        for seq_ix, s in enumerate(seq)
                    ])
                    # Output alignment
                    print native_seq
                    print "".join([" " if c1 != c2 else "|" for c1, c2 in izip(native_seq, seq)])
                    print seq
                    batch_seqs.append(self._seq_to_onehot(seq, length))
                batch_dict["sequences"][:,:,:20] = np.asarray(batch_seqs)

            # Feed dict
            feed_dict.update({
                self.placeholder_sets[i][key]: batch_dict[key] 
                for key in self.feed_keys
            })
        return feed_dict, batch_id_sets

    def sample_batch_test(self, test_code, size=16):
        """ Sample a test batch """
        ids, weights = self.test_train[test_code]
        feed_dict = {}
        batch_id_sets = []
        for i in xrange(self.num_gpus):
            batch_ids = np.random.choice(ids, size, p=weights).tolist()
            batch_ix = [self.fold_ids.index(name) for name in batch_ids]
            batch_dict = self._build_minibatch(batch_ix)
            batch_id_sets.append(batch_ids)
            # Feed dict
            feed_dict.update(
                {self.placeholder_sets[i][key]: batch_dict[key] for key in self.feed_keys}
            )
        return feed_dict, batch_id_sets

    def sample_batch_custom(self, batch_ids):
        """ Sample a custom batch """
        feed_dict = {}
        batch_id_sets = []
        for i in xrange(self.num_gpus):
            batch_ix = [self.fold_ids.index(name) for name in batch_ids]
            batch_dict = self._build_minibatch(batch_ix)
            # Feed dict
            batch_id_sets.append(batch_ids)
            feed_dict.update(
                {self.placeholder_sets[i][key]: batch_dict[key] for key in self.feed_keys}
            )
        return feed_dict, batch_id_sets
