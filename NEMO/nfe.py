import tensorflow as tf
import numpy as np

from physics import Physics
from loss import Loss

class NeuralFoldingEngine:
    def __init__(self, num_gpus=1, max_length=100, batch_size=128, dims={},
                 hyperparameters={}, build_backprop=True, use_profiles=False,
                 use_ss=False, use_ec=False):
        # Model dimensions
        self.dims = {
            "length": max_length,
            "batch": batch_size,
            "alphabet": 20,
            "ss": 8,
            "coarse": 1,
            "atoms": 5
        }
        self.dims["coords"] = 3 * self.dims["atoms"]
        self.use_ec = use_ec
        if use_profiles and not use_ss:
            self.dims["alphabet"] = 40
        elif use_ss:
            self.dims["alphabet"] = 23

        # Model hyperparameters are structured data
        self.hyperparams = {
            "mode": {
                "predict_ss3": False,
                "predict_static": False
            },
            "static": {
                "layers": 2,
                "hidden": 700
            },
            "energy": {
                "cnn": {
                    "out_channels": 128,
                    "conv_hidden": 128,
                    "conv_widths":    [3, 3, 3, 3] * 3,
                    "conv_dilations": [1, 2, 4, 8] * 3
                },
                "graph_conv": {
                    "num_iterations": 3,
                    "channels_1D": 128,
                    "channels_noise": 8,
                    "channels_2D": 50,
                    "conv1D": {
                        "widths": [3, 3, 3, 3],
                        "dilations": [1, 2, 4, 8]
                    },
                    "conv2D": {
                        "widths": [7],
                        "dilations": [1]
                    },
                    "length_scale": 100,
                    "init_sig": {
                        "prior": -2.,
                        "encoder": -3.,
                        "scale": 0.
                    }
                },
                "site_layers": {
                    "conv1D": {
                        "widths": [3, 3, 3, 3] * 3,
                        "dilations": [1, 2, 4, 8] * 3
                    },
                    "width": 11
                },
                "pair_layers": {
                    "width": 9,
                },
                "backbone": {
                    "init": {
                        "scale": 30.0,
                        "length": 7.
                    }
                },
                "angles": {
                    "init": {
                        "scale": 11.0,
                        "angle": 1.2
                    }
                },
                "dihedrals": {
                    "init": {
                        "scale": 20.0,
                        "angle": 0.6
                    }
                },
                "backbone_conv": {
                    "hidden": 64,
                    "width": 11,
                    "lengths": 3, 
                    "scale": 20.,
                    "inflate": 100.,
                    "init_std": 40.
                },
                 "contact_conv": {
                    "hidden": 16,
                    "width": 3,
                    "lengths": 4,
                    "distance": 40
                },
                "mass": {
                    "width": 30,
                    "X": 0.5,
                    "Z_inv": [0.01, 0.01, 0.01],
                    "offD_sigma_init": 1E-4
                },
                "use_max": False,
                "pair_scale_init": 0.6,
                "site_scale_init": 0.2
            },
            "folding": {
                "langevin_steps": 250,
                "relativistic": False,
                "constants": {
                    "rel_m": 1.0,
                    "rel_c": 20.0,
                    "rel_eps": 0.00082,
                    "dist_eps": 0.2,
                    "jacob_eps": 0.05,
                    "xz_eps": 0.01,
                    "angle_eps": 0.1,
                    "alpha": 1.02,
                    "beta": 0.7,
                    "eMx": 0.02,
                    "overlap_dist": 3.0,
                    "overlap_coeff": 0.001,
                    "force_norm_coeff": 0.01,
                    "mass_freeze": 0.333,
                    "speed_cost": 10.
                },
                "init_score": {
                    "v": 0.598,
                    "R0": 2.08,
                    "coeff": 10.0
                }
            },
            "reconstruction": {
                "conv_widths":    [3, 3, 3, 3] * 3,
                "conv_dilations": [1, 2, 4, 8] * 3
            },
            "scoring": {
                "pseudolikelihood": {
                    "importance_samples": 20,
                    "deviation": 4.0,
                    "dist_eps": 0.2,
                    "noise": 0.1
                },
                "logdists": {
                    "eps": 0.2,
                    "radius": 5,
                    "short_prop": 0.5,
                    "contact": 12.,
                    "contact_precision": 5.
                },
                "energy": {
                    "coeff_init": 0.0001,
                    "coeff": 0.0001
                },
                "overlap": {
                    "weight": 1.0
                }
            },
            "training": {
                "dropout": 0.9,
                "learning_rate_init": 0.001,
                "learning_rate_final": 0.0001,
                "learning_rate_time": 65000,
                "generator_slowdown": 1.0,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1E-8,
                "grad_clip_norm": 1.0E2
            },
            "test_keys": [
                "valid_A", "valid_T", "valid_H"
            ]
        }

        self.learning_rates = {
            "": 0.001,
            "lrA": 0.001
        }

        tf.set_random_seed(42)

        with tf.device('/cpu:0'):
            # Setup global optimizer
            self.global_step = tf.Variable(0, name="global_step", trainable=False)

            # BPTT decay
            self.bptt_log_decay = tf.Variable(0., name="bptt_log_decay", trainable=False)
            
            # Each tower has placeholders, tensors, and gradients
            self.placeholder_sets = []
            self.tensor_sets = []
            self.aux_ops = []
            self.bptt_decay_ops = []
            gradient_sets = []

            # Build first tower
            with tf.variable_scope("Towers"):
                for i in xrange(num_gpus):
                    print "Building tower %d" % (i+1)
                    with tf.device("/gpu:%d" % i):
                        with tf.name_scope("Tower%d" % i):
                            placeholders, tensors, gradients, train_ops = \
                                self.build_tower(build_backprop)
                            self.placeholder_sets.append(placeholders)
                            self.tensor_sets.append(tensors)
                            self.aux_ops += train_ops
                            gradient_sets.append(gradients)

                            # Reuse variables from first tower
                            tf.get_variable_scope().reuse_variables()

                            # Set up summaries
                            if i is 0:
                                self.summaries = {}
                                keys = [
                                    "gen_batch", "gen_dense"
                                ] + self.hyperparams["test_keys"]
                                for key in keys:
                                    self.summaries[key] = tf.summary.merge_all(key=key)
            if build_backprop:
                gradients = self.average_gradients(gradient_sets)

        # Add global check numerics ops
        numeric_checks = []
        for v in tf.trainable_variables():
            numeric_checks += [tf.check_numerics(v, v.name + " is not finite")]
        for i in xrange(num_gpus):
            for t in self.tensor_sets[i]:
                if tf.contrib.framework.is_tensor(t):
                    numeric_checks += [tf.check_numerics(t, t.name + " is not finite")]
        self.numeric_check = tf.group(*numeric_checks)

        # Setup learning ops
        if build_backprop:
            with tf.device('/cpu:0'):
                with tf.variable_scope("GeneratorOpt"):
                    # Conditionally execute Adam if BPTT is stable
                    clipped_gvs, self.grad_norm_gen = self.clip_gradients(gradients)
                    train_op = tf.cond(
                        tf.is_finite(self.grad_norm_gen),
                        lambda: self.Adam(gradients, self.global_step),
                        lambda: tf.no_op()
                    )
                    # Always execute BPTT decay and numeric ops
                    train_ops = [train_op] + self.bptt_decay_ops + [self.numeric_check]
                    self.train_gen = tf.group(*train_ops)

    def build_tower(self, build_backprop):
        with tf.variable_scope("Inputs"):
            # Input placeholders [Batch, Length, X]
            placeholders = {}
            placeholders["sequences"] = \
                tf.placeholder(tf.float32, [None, None, self.dims["alphabet"]],
                               name="Sequences")
            placeholders["secondary_structure"] = \
                tf.placeholder(tf.float32, [None, None, self.dims["ss"]],
                               name="SecondaryStructure")
            placeholders["coordinates_target"] = \
                tf.placeholder(tf.float32, [None, None, self.dims["coords"]],
                               name="TargetCoordinates")
            placeholders["lengths"] = \
                tf.placeholder(tf.int32, [None], name="SequenceLengths")
            placeholders["structure_mask"] = \
                tf.placeholder(tf.float32, [None, None], name="StructureMask")

            placeholders["sequences"] = tf.identity(placeholders["sequences"])
            placeholders["bptt_log_decay"] = self.bptt_log_decay

            if self.use_ec:
                placeholders["couplings"] = tf.placeholder(
                    tf.float32, [None, None, None], name="Couplings"
                )

            # Optional placeholders
            default_dropout = tf.constant(
                self.hyperparams["training"]["dropout"], dtype=tf.float32
            )
            placeholders["global_step"] = self.global_step
            placeholders["training"] = tf.placeholder_with_default(
                False, (), name="Training"
            )
            placeholders["dropout"] = tf.placeholder_with_default(
                default_dropout, (), name="Dropout"
            )
            placeholders["native_init_prob"] = tf.placeholder_with_default(
                0., (), name="NativeInitProb"
            )
            placeholders["native_unfold_max"] = tf.placeholder_with_default(
                0.5, (), name="NativeUnfoldMax"
            )
            placeholders["native_unfold_randomize"] = tf.placeholder_with_default(
                False, (), name="NativeRandomUnfold"
            )
            placeholders["langevin_steps"] = tf.placeholder_with_default(
                self.hyperparams["folding"]["langevin_steps"], (), name="LangevinSteps"
            )
            placeholders["beta_anneal"] = tf.placeholder_with_default(
                1.0, (), name="BetaAnneal"
            )
            # Update unknown dims
            dims = self.dims
            seq_shape = tf.shape(placeholders["sequences"])
            dims["batch"] = seq_shape[0]
            dims["length"] = seq_shape[1]

        # Keep track of which variables we create
        prior_vars = tf.trainable_variables()

        # Build the protein folding engine
        physics = Physics(placeholders, dims, self.global_step, self.hyperparams)
        physics.build_graph()
        tensors = physics.tensors

        # Build the loss function 
        loss = Loss(placeholders, tensors, dims, self.global_step, self.hyperparams)
        loss.build_graph()
        tensors["score"] = loss.tensors["score"]
        tensors["coarse_target"] = loss.tensors["coarse_target"]

        new_vars = tf.trainable_variables()
        params = {}
        params["score"] = [v for v in new_vars if "Score" in v.name]
        params["generator"] = [
            v for v in new_vars if v not in params["score"]
        ]

        # Parameter accounting
        for (param_name, param_set) in params.iteritems():
            print "Parameter set: " + param_name
            p_counts = [np.prod(v.get_shape().as_list()) for v in param_set]
            p_names = [v.name for v in param_set]
            p_total = sum(p_counts)
            print "    contains " + str(p_total) + " params in " \
                + str(len(p_names)) + " tensors"
            print "\n"

        # Backpropagate
        gradients = None
        train_ops = physics.layers.train_ops + loss.layers.train_ops
        if build_backprop:
            with tf.variable_scope("Backprop"):
                with tf.variable_scope("Generator"):
                    g_loss = tensors["score"]
                    g_vars = params["generator"] + params["score"]
                    g_grads = tf.gradients(g_loss, g_vars)
                    gen_gvs = zip(g_grads, g_vars)
                    gradients, gen_grad_norm = self.clip_gradients(gen_gvs)

                    """ Gradient damping (gamma) adaptation for simulator

                        During simulator roll-outs, backpropgation is damped as
                        gamma = exp(log_decay)
                        X = gamma * X + (1. - gamma) * tf.stop_gradient(X)

                        When gamma=1.0, this behaves in the usual manner.

                        [ Current Heuristic ]
                        When the model is 'stable':
                            slowly raise gamma with log-geometric scaling
                        When the model is 'unstable':
                            rapidly lower gamma with log-linear scaling
                    """
                    is_stable = tf.logical_and(
                        tf.is_finite(gen_grad_norm), gen_grad_norm < 100.
                    )
                    decay_update = tf.cond(
                        is_stable,
                        lambda: 0.99 * self.bptt_log_decay,
                        lambda: self.bptt_log_decay - 0.01
                    )
                    self.bptt_decay_ops += [self.bptt_log_decay.assign(decay_update)]

                    tf.summary.scalar("GradientNorm", gen_grad_norm,
                              collections=["gen_batch", "gen_dense"])

                    tf.summary.scalar("BPTTLogDecay", self.bptt_log_decay,
                              collections=["gen_batch", "gen_dense"])

        return placeholders, tensors, gradients, train_ops

    def average_gradients(self, gradient_set):
        """ Average the gradient sets """
        averaged_gv = []
        if len(gradient_set) > 1:
            for gv in zip(*gradient_set):
                g_set = [g for (g,v) in gv]
                g_mean = tf.reduce_mean(tf.stack(g_set, axis=0), 0)
                v_mean = gv[0][1]
                averaged_gv.append((g_mean, v_mean))
        else:
            averaged_gv = gradient_set[0]
        return averaged_gv

    def clip_gradients(self, gvs):
        """ Clip the gradients """
        grads, gvars = list(zip(*gvs)[0]), list(zip(*gvs)[1])
        clip_norm = self.hyperparams["training"]["grad_clip_norm"]
        clipped_grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
        clipped_gvs = zip(clipped_grads, gvars)
        return clipped_gvs, global_norm

    def Adam(self, gvs, global_step):
        """ Build Adam optimizer """
        _dtype = tf.float32

        with tf.variable_scope("Adam"):
            # Build moving averages and updates
            grad_list, var_list = list(zip(*gvs)[0]), list(zip(*gvs)[1])
            _z = lambda v, n: tf.Variable(
                np.zeros(v.get_shape().as_list()), name=n, 
                dtype=_dtype, trainable=False
            )
            with tf.variable_scope("Averages"):
                m_set = [_z(v, "m") for v in var_list]
                sq_set = [_z(v, "sq") for v in var_list]

            # Build Adam updates
            # t <- t + 1
            # lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
            # m_t <- beta1 * m_{t-1} + (1 - beta1) * g
            # v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
            # variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)

            beta_1 = self.hyperparams["training"]["beta_1"]
            beta_2 = self.hyperparams["training"]["beta_2"]
            m_updates = [
                m.assign(beta_1 * m + (1. - beta_1) * g)
                for (m, g) in zip(m_set, grad_list)
            ]
            sq_updates = [
                sq.assign(beta_2 * sq + (1. - beta_2) * g * g)
                for (sq, g) in zip(sq_set, grad_list)
            ]
            avg_updates = m_updates + sq_updates

            # Time updates
            t_update = global_step.assign(global_step + 1)
            t = tf.cast(t_update, _dtype)
            lr_t = tf.sqrt(1. - tf.pow(beta_2, t))\
                    / (1. - tf.pow(beta_1, t))

            # Learning rates based on name keys
            # lr_list = []
            # for v in var_list:
            #     rates = [
            #         rate for key, rate
            #         in self.learning_rates.iteritems()
            #         if key in v.name
            #     ]
            #     lr_list.append(max(rates))
            lr = tf.cond(
                global_step < self.hyperparams["training"]["learning_rate_time"],
                lambda: self.hyperparams["training"]["learning_rate_init"],
                lambda: self.hyperparams["training"]["learning_rate_final"]
            )

            epsilon = self.hyperparams["training"]["epsilon"]
            v_updates = [
                v.assign(v - lr * lr_t * m / (tf.sqrt(sq) + epsilon))
                for v, m, sq in zip(var_list, m_updates, sq_updates)
            ]
            v_updates += self.aux_ops
            opt_op = tf.group(*v_updates)
        return opt_op

    def opt_op(self, optimizer, gvs, global_step=None):
        """ NaN-gated optimization step """
        gvs_clipped, grad_norm = self.clip_gradients(gvs)
        if global_step is not None:
            opt_op = tf.cond(
                tf.is_finite(grad_norm),
                lambda: optimizer.apply_gradients(gvs_clipped, global_step=global_step),
                lambda: tf.no_op()
            )
        else:
            opt_op = tf.cond(
                tf.is_finite(grad_norm),
                lambda: optimizer.apply_gradients(gvs_clipped),
                lambda: tf.no_op()
            )
        return opt_op, grad_norm
