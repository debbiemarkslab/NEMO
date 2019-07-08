import tensorflow as tf
import numpy as np
from layers import Layers

class Physics:
    def __init__(self, placeholders={}, dims={}, global_step=0, hyperparams={}):
        self.dims = dims
        self.hyperparams = hyperparams
        self.placeholders = placeholders
        self.global_step = global_step
        self.tensors = {}
        return

    def build_graph(self):
        """Build the computational graph for the entire physics module
        """
        self.layers = Layers(self.placeholders)
        self.masks = self._build_masks()

        # Coarse target
        self.tensors["coordinates_target"], self.tensors["coarse_target"], \
            self.tensors["coarse_target_logD"], self.tensors["coarse_contacts"] = \
            self._build_target()

        if self.hyperparams["mode"]["predict_static"]:
            # Build a static predictor rather than physics simulation
            self.tensors["coordinates_coarse"]  = self._build_static_predictor()

            self.tensors["trajectory"] = \
                tf.expand_dims(self.tensors["coordinates_coarse"], 1)
            self.tensors["trajectory_logprobs"] = tf.zeros([self.dims["batch"], 1])
            self.tensors["loss_forcefield"] = 0
            self.tensors["loss_init"] = 0
            self.tensors["loss_dynamics"] = 0

        else:
            # Energy function
            self.tensors["energy"], self.tensors["loss_forcefield"] = \
                self._build_energy()

            # Folding simulation
            self.tensors["coordinates_coarse"], self.tensors["trajectory"], \
                self.tensors["trajectory_logprobs"], self.tensors["loss_init"], \
                self.tensors["loss_dynamics"] = self._build_dynamics()

        # Atomic granulation
        self.tensors["coordinates_fine"], self.tensors["SS"], \
            self.tensors["loss_reconstruction"] = self._build_atomizer()

        # Weight regularization
        self.tensors["loss_weights_physics"] = self.layers.loss
        return

    def _build_masks(self):
        """Build masks
        Returns:
        """
        lengths = self.placeholders["lengths"]
        structure_mask = self.placeholders["structure_mask"]
        with tf.variable_scope("Masks"):
            masks = {}
            # Build mask
            lengths_tiled = tf.tile(tf.expand_dims(lengths, 1),
                                    [1, self.dims["length"]])
            indices = tf.expand_dims(tf.range(0, self.dims["length"], 1), 0)
            indices_tiled = tf.tile(indices, [self.dims["batch"], 1])
            masks["indices"] = tf.to_float(indices_tiled)
            # 2D distances (not a mask)
            masks["ij_dist"] = tf.to_float(tf.abs(
                tf.expand_dims(indices_tiled, 2)
                - tf.expand_dims(indices_tiled, 1)
            ))
            # Boolean tensor (batch, length)
            mask = tf.less(indices_tiled, lengths_tiled)
            # Ones tensor (batch, length)
            masks["seqs"] = tf.where(mask,
                                     tf.ones([self.dims["batch"],
                                              self.dims["length"]]),
                                     tf.zeros([self.dims["batch"],
                                               self.dims["length"]]))
            # Ones tensor (Batch, 1, length)
            masks["forces"] = tf.expand_dims(masks["seqs"], 1)
            # (N,L,L)
            off_diag = 1. - \
                tf.expand_dims(tf.diag(tf.ones([self.dims["length"]])), 0)
            masks["2D"] = tf.expand_dims(
                masks["seqs"], 1) * tf.expand_dims(masks["seqs"], 2)
            masks["conv1D"] = tf.reshape(
                masks["seqs"],
                [self.dims["batch"], 1, self.dims["length"], 1]
            )
            masks["conv2D"] = tf.expand_dims(masks["2D"], 3)
            masks["dists"] = masks["2D"] * off_diag
            # (N,1,L,L)
            masks["resids"] = tf.expand_dims(masks["dists"], 1)
            # (N,L*)
            masks["backbone"] = tf.slice(masks["seqs"], [0, 1], [-1, -1])
            masks["angles"] = tf.slice(masks["seqs"], [0, 2], [-1, -1])
            masks["dihedrals"] = tf.slice(masks["seqs"], [0, 3], [-1, -1])
            # (N,L,1,1) for internal coordinate forces
            masks["dL"] = tf.pad(masks["backbone"], [
                                 [0, 0], [1, 0]], "CONSTANT")
            masks["dT"] = tf.pad(masks["angles"], [[0, 0], [2, 0]], "CONSTANT")
            masks["dP"] = tf.pad(masks["dihedrals"], [
                                 [0, 0], [3, 0]], "CONSTANT")
            f = lambda X: tf.reshape(
                X, [self.dims["batch"], self.dims["length"], 1, 1]
            )
            masks["dL"] = f(masks["dL"])
            masks["dT"] = f(masks["dT"])
            masks["dP"] = f(masks["dP"])
            # Mask for internal coordinates
            masks["conv1D_Z"] = tf.pad(
                masks["conv1D"][:,:,3:,:], [[0,0],[0,0],[3,0],[0,0]], "CONSTANT"
            )
            # Structure masks
            base_mask = masks["seqs"] * structure_mask
            # (N, L, L)
            masks["structure_coarse_dists"] = tf.expand_dims(
                base_mask, 1) * tf.expand_dims(base_mask, 2) * off_diag
            # (N, NUM_ATOMS * L, NUM_ATOMS * L)
            L_expand = self.dims["length"] * self.dims["atoms"]
            mask_expanded = tf.tile(
                tf.expand_dims(base_mask, 2), [1, 1, self.dims["atoms"]])
            mask_expanded = tf.reshape(
                mask_expanded, [self.dims["batch"], L_expand])
            off_diag_expand = 1. - \
                tf.expand_dims(tf.diag(tf.ones([L_expand])), 0)
            masks["structure_fine_dists"] = tf.expand_dims(mask_expanded, 1) \
                * tf.expand_dims(mask_expanded, 2) * off_diag_expand
            # (N,L)
            masks["ss"] = masks["seqs"] * structure_mask
            # Backbone structure
            masks["structure"] = self.placeholders["structure_mask"]
            masks["structure_backbone"] = \
                tf.slice(masks["structure"], [0, 1], [-1, self.dims["length"] - 1]) \
                * tf.slice(masks["structure"], [0, 0], [-1, self.dims["length"] - 1])
            # Mask the structure
            left_mask = masks["structure"][:, 0:self.dims["length"]-1]
            right_mask = masks["structure"][:, 1:self.dims["length"]]
            left_mask_pad = tf.pad(left_mask, [[0, 0], [1, 0]], "CONSTANT")
            right_mask_pad = tf.pad(right_mask, [[0, 0], [0, 1]], "CONSTANT")
            masks["structure_internals"] = \
                masks["structure"] * left_mask_pad * right_mask_pad
            # Structure Convolution masks
            masks["structure_1D"] = tf.expand_dims(
                tf.expand_dims(masks["structure"], 1), 3
            )
            masks["structure_2D"] = tf.expand_dims(
                tf.expand_dims(base_mask, 1) * tf.expand_dims(base_mask, 2), 3
            )
            # Jacobian mask is (B,L,L,1) for (B,i,j,xyz)
            # Lower triangle (inclusive)
            ix = tf.range(0, self.dims["length"], 1)
            i_tiled = tf.tile(tf.expand_dims(ix, 1), [1, self.dims["length"]])

            j_tiled = tf.transpose(i_tiled)
            i_leq_j = tf.less_equal(i_tiled, j_tiled)
            i_leq_j_mask = tf.where(
                i_leq_j,
                tf.ones([self.dims["length"], self.dims["length"]]),
                tf.zeros([self.dims["length"], self.dims["length"]])
            )
            # (L,L) => (B,L,L,1) with length masking
            masks["jacobian"] = tf.expand_dims(
                tf.expand_dims(i_leq_j_mask, 0), 3
            )
            # Short and long split
            ij_cutoff = self.hyperparams["energy"]["backbone_conv"]["width"]
            masks["ij_short"] = masks["dists"] * tf.where(
                        masks["ij_dist"] <= ij_cutoff,
                        tf.ones_like(masks["ij_dist"]),
                        tf.zeros_like(masks["ij_dist"])
            )
            masks["ij_long"] = masks["dists"] * (1. - masks["ij_short"])
            masks["ij_long_x4"] = tf.nn.avg_pool(
                value=tf.expand_dims(masks["ij_long"],3), 
                ksize=[1, 1, 1, 1], 
                strides=[1, 4, 4, 1],
                padding="SAME"
            )
            # For structure
            masks["ij_short_struct"] = masks["structure_coarse_dists"] * tf.where(
                        masks["ij_dist"] <= ij_cutoff,
                        tf.ones_like(masks["ij_dist"]),
                        tf.zeros_like(masks["ij_dist"])
            )
            masks["ij_long_struct"] = masks["structure_coarse_dists"] * \
                (1. - masks["ij_short"])
        return masks

    def _build_static_predictor(self):
        """Build the computational graph for a static structure prediction. """

        def _positive_cons(X):
            return tf.nn.softplus(X)

        def _inv_positive_cons(Y):
            return np.log(np.exp(Y) - 1.0)

        def _cart_to_spherical(dX, mask):
            """ Convert Cartesian vector into spherical coordinates """
            l_eps = 0.001
            dx, dy, dz = tf.unstack(dX, axis=1)
            sq_yz  = tf.square(dy) + tf.square(dz) 
            sq_xyz = sq_yz + tf.square(dx)
            L_yz = tf.sqrt(sq_yz + l_eps)
            L = tf.sqrt(sq_xyz + l_eps)
            # T is angle between of u([dx,dy,dz]) and [1,0,0]
            T = mask * tf.acos(dx / L)
            # P is angle between u([0,dy,dz]) and [0,1,0]
            P = mask * tf.acos(dy / L_yz) * tf.sign(dz)
            return L, T, P

        def _linear_interface(X, name, mask, mu_init, sigma_init,
                              positive=False):
            """Distribute rescaled layer around a global bias
            """
            if positive:
                if _inv_positive_cons(mu_init) > 0.:
                    sigma_init *= mu_init
                mu_init = _inv_positive_cons(mu_init)
            with tf.variable_scope(name):
                # Mean of the distribution
                mu = tf.get_variable(
                    "mu_lrA", (),
                    initializer=tf.constant_initializer(mu_init)
                )
                # Standard deviation of the distribution
                sigma_init = _inv_positive_cons(sigma_init)
                sigma = tf.get_variable(
                    "sigma_lrA", (),
                    initializer=tf.constant_initializer(sigma_init)
                )
                sigma = _positive_cons(sigma)
                rank = len(X.get_shape().as_list())
                with tf.variable_scope("Loss"):
                    if "loss_interfaces" not in self.tensors:
                        self.tensors["loss_interfaces"] = 0.
                    if mask is None:
                            mask = tf.ones_like(X)
                    else:
                        mask = mask * tf.ones_like(X)
                    X_mean = tf.reduce_sum(mask * X) \
                            / tf.reduce_sum(mask)
                    X_residual = X-X_mean
                    X_std = tf.sqrt(
                        tf.reduce_sum(mask * tf.square(X_residual))
                        / tf.reduce_sum(mask) + 1E-5
                    )
                    self.tensors["loss_interfaces"] += (
                        tf.square(X_mean) + tf.square(X_std - 1.)
                    )
                    # X = tf.clip_by_value(X, -3., 3.)
                    X = 3. * tf.tanh(0.333 * X)
                with tf.variable_scope("Rescale"):
                    X = sigma * X + mu
                    if positive:
                        X = _positive_cons(X)
                    if mask is not None:
                        X *= mask
            return X

        with tf.variable_scope("StaticInitializer"):
            with tf.variable_scope("SequenceFeatures"):
                seqs = self.placeholders["sequences"]
                seqs_1D = tf.expand_dims(seqs, 1)
                # Upscale sequence encodings to have raw second moment = 1
                if self.dims["alphabet"] == 20:
                    seq_scale = np.sqrt(self.dims["alphabet"])
                    seqs_1D *= seq_scale
                else:
                    seqs_1D = self.layers.normalize_layer(
                        seqs_1D, self.masks["conv1D"], name="NormSeqs"
                    )
                self.tensors["seqs_1D"] = seqs_1D

            with tf.variable_scope("RNN"):
                rnn_layers = self.hyperparams["static"]["layers"]
                rnn_hidden = self.hyperparams["static"]["hidden"]
                up_rnn = tf.squeeze(seqs_1D, 1)
                for i in xrange(rnn_layers):
                    (out_fw, out_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        tf.nn.rnn_cell.LSTMCell(rnn_hidden), 
                        tf.nn.rnn_cell.LSTMCell(rnn_hidden), 
                        up_rnn, dtype=tf.float32,
                        sequence_length=self.placeholders["lengths"],
                        scope="RNN" + str(i)
                    )
                    up_rnn = tf.concat(axis=2, values = [out_fw, out_bw])
                    up_rnn = tf.nn.dropout(up_rnn, self.placeholders["dropout"])
                    # up_rnn = tf.squeeze(layers.normalize_layer(
                    #     tf.expand_dims(up_rnn, 1),
                    #     self.masks["conv1D"], per_channel=False
                    # ), 1)
                up_rnn = tf.expand_dims(up_rnn, 1)
            
            with tf.variable_scope("Predictions"):
                # Outputs
                out_dims = 4 + self.dims["ss"]
                outputs = self.layers.conv2D(
                    up_rnn,
                    filters=out_dims, kernel_size=[1, 1],
                    padding="same", activation=None,
                    use_bias=True, name="Outputs",
                    batchnorm=False, mask=self.masks["conv1D"]
                )

                # Predict Cartesian vectors for angles only
                dX_1_unc = outputs[:,0,:,0]
                dX_2_unc = outputs[:,0,:,1]
                dX_3_unc = outputs[:,0,:,2]
                L_init_unc = outputs[:,0,:,3]
                self.tensors["ss_pre_logits"] = outputs[:,0,:,4:]

                # Parameterize lengths with an exponential for fast changes
                hyperparams = self.hyperparams["energy"]
                L_init = tf.expand_dims(_linear_interface(
                    L_init_unc, "length_init", self.masks["seqs"],
                    hyperparams["backbone"]["init"]["length"],
                    hyperparams["site_scale_init"],
                    positive=True
                ), 1)

                # Build bias
                bias_T = hyperparams["angles"]["init"]["angle"]
                bias_P = hyperparams["dihedrals"]["init"]["angle"]
                dX_1 = _linear_interface(
                    dX_1_unc, "dX1", self.masks["seqs"],
                    np.cos(bias_T),
                    hyperparams["site_scale_init"],
                    positive=False
                )
                dX_2 = _linear_interface(
                    dX_2_unc, "dX2", self.masks["seqs"],
                    np.sin(bias_T) * np.cos(bias_P),
                    hyperparams["site_scale_init"],
                    positive=False
                )
                dX_3 = _linear_interface(
                    dX_3_unc, "dX3", self.masks["seqs"],
                    np.sin(bias_T) * np.sin(bias_P),
                    hyperparams["site_scale_init"],
                    positive=False
                )
                dX = tf.stack([dX_1,dX_2,dX_3], axis=1)
                dX = L_init * tf.nn.l2_normalize(dX, 1)
                L_init, T_init, P_init = _cart_to_spherical(dX, self.masks["seqs"])
                X_coarse = self._dynamics_to_cartesian(L_init, T_init, P_init)
                

            # with tf.variable_scope("Predictions"):
            #     # Outputs
            #     out_dims = 4 + self.dims["ss"]
            #     outputs = self.layers.conv2D(
            #         up_rnn,
            #         filters=out_dims, kernel_size=[1, 1],
            #         padding="same", activation=None,
            #         use_bias=True, name="Outputs",
            #         batchnorm=False, mask=self.masks["conv1D"]
            #     )

            #     # Separately decompose direction and length
            #     ux = tf.nn.l2_normalize(outputs[:,0,:,:3],2)
            #     l = tf.nn.softplus(outputs[:,0,:,3])
            #     self.tensors["ss_pre_logits"] = outputs[:,0,:,4:]
            #     dX = tf.expand_dims(l,2) * ux
            #     L_init, T_init, P_init = _cart_to_spherical(dX, self.masks["seqs"])
            #     X_coarse = self._dynamics_to_cartesian(L_init, T_init, P_init)

        return X_coarse

    def _build_energy(self):
        """Build the computational graph for the energy function.

        For 2D restraints (pair potentials), a multilayer perceptron is applied
        to all possible pair combinations of 1D latent features.
        Optionally this block of 2D parameters can be postprocessed by a 2D
        convolutional neural network.

        For 1D restraints (backbone and angle potentials), a 1D convolutional
        network is applied to the 1D latent features.

        Restraints:
            Pair potentials (2D):
                SoftPlus6 (rep, loc, radius, depth, ramp, attr)
            Pair potentials (1D):
                Contact potential
                Repulsion potential

            Backbone restraints
                Harmonic bond lengths (scale, )
                Angles
                Dihedral angles

        Args:
            latent: The 1D sequence features generated by the sequence encoder
                with dimensionality [batch_size, max_length, num_letters].

            hyperparameters: A dictionary containing hyperparameters for the
                various networks used in force field construction.

        Returns:
            A dictionary of different kinds of force field restraints, each of
            which contains a list of sets of parameters. This allows each type
            of force field term to be replicated multiple times if desired.
        """

        def _positive_cons(X):
            return tf.nn.softplus(X)

        def _inv_positive_cons(Y):
            return np.log(np.exp(Y) - 1.0)

        def _safe_summary(name, X):
            X_safe = tf.where(tf.is_nan(X), -10 * tf.ones_like(X), X)
            tf.summary.histogram(name, X_safe,
                                 collections=["gen_dense"])
            if len(X.get_shape().as_list()) < 4:
                X = tf.expand_dims(X, 3)
            tf.summary.image(name, X, collections=["gen_dense"])
            return

        def _linear_interface(X, name, mask, mu_init, sigma_init,
                              positive=False):
            """Distribute rescaled layer around a global bias
            """
            if positive:
                if _inv_positive_cons(mu_init) > 0.:
                    sigma_init *= mu_init
                mu_init = _inv_positive_cons(mu_init)
            with tf.variable_scope(name):
                # Mean of the distribution
                mu = tf.get_variable(
                    "mu_lrA", (),
                    initializer=tf.constant_initializer(mu_init)
                )
                # Standard deviation of the distribution
                sigma_init = _inv_positive_cons(sigma_init)
                sigma = tf.get_variable(
                    "sigma_lrA", (),
                    initializer=tf.constant_initializer(sigma_init)
                )
                sigma = _positive_cons(sigma)
                rank = len(X.get_shape().as_list())
                with tf.variable_scope("Loss"):
                    if "loss_interfaces" not in self.tensors:
                        self.tensors["loss_interfaces"] = 0.
                    if mask is None:
                            mask = tf.ones_like(X)
                    else:
                        mask = mask * tf.ones_like(X)
                    X_mean = tf.reduce_sum(mask * X) \
                            / tf.reduce_sum(mask)
                    X_residual = X-X_mean
                    X_std = tf.sqrt(
                        tf.reduce_sum(mask * tf.square(X_residual))
                        / tf.reduce_sum(mask) + 1E-5
                    )
                    self.tensors["loss_interfaces"] += (
                        tf.square(X_mean) + tf.square(X_std - 1.)
                    )
                    # X = tf.clip_by_value(X, -3., 3.)
                    X = 3. * tf.tanh(0.333 * X)
                with tf.variable_scope("Rescale"):
                    X = sigma * X + mu
                    if positive:
                        X = _positive_cons(X)
                    if mask is not None:
                        X *= mask
            return X

        def _field_layer(layers, index):
            layer = tf.squeeze(tf.slice(layers,
                                        [0, index, 0, 0],
                                        [-1, 1, -1, -1]),
                               axis=1
                               )
            layer = layer + tf.transpose(layer, [0,2,1])
            return layer

        def _site_layer(layers, index, channels=1):
            layer = tf.slice(layers, [0, index, 0], [-1, channels, -1])
            if channels == 1:
                layer = tf.squeeze(layer, axis=1)
            return layer

        def _filter_bias(name, height, width, num_in, num_out):
            W_unc = tf.get_variable(
                "W_unc" + name, shape=[height, width, num_in, num_out], 
                dtype=tf.float32, 
                initializer=tf.random_normal_initializer(stddev = 0.01),
                trainable=True
            )
            W_unc = tf.nn.l2_normalize(W_unc, [0,1,2])
            W = W_unc * tf.exp(tf.get_variable(
                "g" + name, shape=[1, 1, 1, num_out],
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
                trainable=True
            ))
            b = tf.get_variable(
                "b" + name, shape=[1, 1, 1, num_out],
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
                trainable=True
            )
            return W, b

        def _unit_vecs(dX):
            """ Normalize vectors """
            l_eps = 1E-3
            L = tf.sqrt(tf.reduce_sum(tf.square(dX), 1, keep_dims=True) + l_eps)
            return dX / L

        def _cart_to_spherical(dX, mask):
            """ Convert Cartesian vector into spherical coordinates """
            l_eps = 0.001
            dx, dy, dz = tf.unstack(dX, axis=1)
            sq_yz  = tf.square(dy) + tf.square(dz) 
            sq_xyz = sq_yz + tf.square(dx)
            L_yz = tf.sqrt(sq_yz + l_eps)
            L = tf.sqrt(sq_xyz + l_eps)
            # T is angle between of u([dx,dy,dz]) and [1,0,0]
            T = mask * tf.acos(dx / L)
            # P is angle between u([0,dy,dz]) and [0,1,0]
            P = mask * tf.acos(dy / L_yz) * tf.sign(dz)
            return L, T, P

        hyperparams = self.hyperparams["energy"]
        with tf.variable_scope("Energy"):
            pairwise_shape = (
                self.dims["batch"], self.dims["length"], self.dims["length"]
            )

            with tf.variable_scope("GraphConv"):
                channels_1D = hyperparams["graph_conv"]["channels_1D"]
                channels_noise = hyperparams["graph_conv"]["channels_noise"]
                channels_2D = hyperparams["graph_conv"]["channels_2D"]
                num_steps = hyperparams["graph_conv"]["num_iterations"]
                widths_1D = hyperparams["graph_conv"]["conv1D"]["widths"]
                dilations_1D = hyperparams["graph_conv"]["conv1D"]["dilations"]
                widths_2D = hyperparams["graph_conv"]["conv2D"]["widths"]
                dilations_2D = hyperparams["graph_conv"]["conv2D"]["dilations"]
                cnn_out_channels = hyperparams["cnn"]["out_channels"]

                # Build matrix of primary distances
                with tf.variable_scope("DistanceFeatures"):
                    # Concatenate distance feature
                    length_scale = hyperparams["graph_conv"]["length_scale"]
                    Dij_normed = self.masks["ij_dist"]
                    num_rbfs = 5
                    sigma = 0.5 * length_scale / num_rbfs
                    dist_layer = tf.expand_dims(Dij_normed, 3)
                    ij_rbfs = [
                        tf.exp(-tf.square((dist_layer - c) / sigma))
                        for c in np.linspace(0, length_scale, num=num_rbfs)
                    ] + [dist_layer  / length_scale]
                    dist_layer = tf.concat(
                        axis = 3, 
                        values = ij_rbfs + [
                            tf.expand_dims(self.masks["ij_short"], 3),
                            tf.expand_dims(self.masks["ij_long"], 3)
                        ]
                    )
                    dist_layer = self.layers.normalize_layer(
                        dist_layer, self.masks["conv2D"], name="A"
                    )
                with tf.variable_scope("SequenceFeatures"):
                    seqs = self.placeholders["sequences"]
                    if self.hyperparams["mode"]["predict_ss3"]:
                        seqs = tf.concat(
                            axis=2, values=[seqs, self.tensors["SS_3_input"]]
                        )
                    seqs_1D = tf.expand_dims(seqs, 1)
                    # Tiled and concatenated version
                    seqs_i = tf.tile(
                        tf.expand_dims(seqs, 1), [1, self.dims["length"], 1, 1]
                    )
                    seqs_j = tf.tile(
                        tf.expand_dims(seqs, 2), [1, 1, self.dims["length"], 1]
                    )
                    seqs_2D = tf.concat(axis = 3, values = [seqs_i, seqs_j])
                    # Upscale sequence encodings to have raw second moment = 1
                    if self.dims["alphabet"] == 20:
                        seq_scale = np.sqrt(self.dims["alphabet"])
                        seqs_1D *= seq_scale
                        seqs_2D *= seq_scale
                    else:
                        seqs_1D = self.layers.normalize_layer(
                            seqs_1D, self.masks["conv1D"], name="NormSeqs"
                        )
                    self.tensors["seqs_1D"] = seqs_1D

                with tf.variable_scope("CNNFeatures"):
                    up_cnn = self.layers.conv2D(
                        seqs_1D, name="upmix",
                        filters=hyperparams["cnn"]["conv_hidden"], 
                        kernel_size=(1, 1),
                        activation=self.layers.nonlinearity, padding="same",
                        batchnorm=True,
                        mask=self.masks["conv1D"]
                    )
                    cnn_1D = self.layers.convnet_1D(
                        up_cnn, hyperparams["cnn"]["conv_hidden"], 
                        self.masks["conv1D"], 
                        hyperparams["cnn"]["conv_widths"], 
                        hyperparams["cnn"]["conv_dilations"],
                        self.placeholders["dropout"]
                    )
                    cnn_1D = self.layers.normalize_layer(
                        cnn_1D, self.masks["conv1D"], name="A"
                    )
                    cnn_1D = self.layers.conv2D(
                        cnn_1D, name="CNN1D",
                        filters=cnn_out_channels, kernel_size=(1, 1),
                        activation=self.layers.nonlinearity, padding="same",
                        batchnorm=True,
                        mask=self.masks["conv1D"]
                    ) * self.masks["conv1D"]
                    cnn_1D = self.layers.normalize_layer(
                        cnn_1D, self.masks["conv1D"], name="B"
                    )
                    # Tiled and concatenated version
                    cnn_base = tf.squeeze(cnn_1D, 1)
                    cnn_i = tf.tile(
                        tf.expand_dims(cnn_base, 1), [1, self.dims["length"], 1, 1]
                    )
                    cnn_j = tf.tile(
                        tf.expand_dims(cnn_base, 2), [1, 1, self.dims["length"], 1]
                    )
                    cnn_2D = tf.concat(axis = 3, values = [cnn_i, cnn_j])

                # Initial state
                with tf.variable_scope("DecoderInit"):
                    features_1D = tf.concat(
                        axis=3, values=[seqs_1D, cnn_1D]
                    )
                    # Transform + Normalize
                    state_1D = self.layers.conv2D(
                        features_1D, name="Features1D",
                        filters=channels_1D, kernel_size=(1, 1),
                        activation=self.layers.nonlinearity, padding="same",
                        batchnorm=True,
                        mask=self.masks["conv1D"]
                    ) * self.masks["conv1D"]
                    # Initial feature processing
                    state_1D = self.layers.convnet_1D(
                        state_1D, channels_1D, self.masks["conv1D"],
                        widths_1D,
                        dilations_1D,
                        self.placeholders["dropout"]
                    )
                    features_2D = tf.concat(
                        axis = 3,
                        values=[dist_layer, seqs_2D, cnn_2D]
                    )
                    if "couplings" in self.placeholders:
                        _safe_summary(
                            "couplings", self.placeholders["couplings"]
                        )
                        couplings_2D = tf.expand_dims(
                            self.placeholders["couplings"], 3
                        )
                        couplings_2D = self.layers.conv2D(
                            couplings_2D, name="Couplings2D",
                            filters=3, kernel_size=(1, 1),
                            activation=tf.nn.sigmoid, 
                            padding="same",
                            batchnorm=True,
                            mask=self.masks["conv2D"]
                        )
                        features_2D = tf.concat(
                            axis = 3, values = [features_2D, couplings_2D]
                        )
                    state_2D = self.layers.conv2D(
                        features_2D, filters=channels_2D,
                        kernel_size=[1, 1], padding="same",
                        activation=self.layers.nonlinearity,
                        batchnorm=True,
                        mask=self.masks["conv2D"]
                    ) * self.masks["conv2D"]
                    # Normalize initialization of network
                    state_1D = self.layers.normalize_layer(
                        state_1D, self.masks["conv1D"], name="1D"
                    )
                    state_2D = self.layers.normalize_layer(
                        state_2D, self.masks["conv2D"], name="2D"
                    )
                    decoder = {"1D": [state_1D], "2D": [state_2D]}
                with tf.variable_scope("DecoderSteps", reuse=tf.AUTO_REUSE):
                    for i in xrange(num_steps):
                        # Graph convolution block
                        with tf.variable_scope("Step" + str(i+1)):
                            reuse = None
                        # with tf.name_scope("Step" + str(i+1)):
                        #     reuse = True if i > 0 else None
                            state_1D, state_2D = self.layers.conv_graph(
                                decoder["1D"][i], decoder["2D"][i],
                                channels_1D, channels_2D, widths_1D,
                                dilations_1D, widths_2D, dilations_2D,
                                self.masks["conv1D"], self.masks["conv2D"],
                                reuse=reuse,
                                dropout_p=self.placeholders["dropout"],
                                aux_1D=features_1D, aux_2D=features_2D, 
                            )
                            decoder["1D"].append(state_1D)
                            decoder["2D"].append(state_2D)

                        with tf.variable_scope("Step" + str(i+1) + "Summaries"):
                            # Representations
                            tf.summary.histogram("Decoder1D",
                                                 decoder["1D"][i + 1],
                                                 collections=["gen_dense"])
                            tf.summary.histogram("Decoder2D",
                                                 decoder["2D"][i + 1],
                                                 collections=["gen_dense"])
                with tf.variable_scope("Nodes"):
                    state_1D = self.layers.normalize_layer(
                        state_1D, self.masks["conv1D"],
                        name="A"
                    )
                    self.tensors["nodes"] = tf.concat(
                        axis=3, values=[features_1D, state_1D]
                    ) * self.masks["conv1D"]
                    self.tensors["edges"] = state_2D
                    # Summary of sequence features
                    nodes_image = tf.expand_dims(tf.transpose(
                            tf.reshape(self.tensors["nodes"],
                                [self.dims["batch"], self.dims["length"], -1]), 
                            perm=[0, 2, 1]
                        ), 3)
                    tf.summary.image(
                        "nodes", nodes_image,
                        collections=["gen_dense"]
                    )

            with tf.variable_scope("PairLayers"):
                num_pair_layers = 3 + hyperparams["contact_conv"]["hidden"] \
                    + hyperparams["contact_conv"]["lengths"]
                pair_width = hyperparams["pair_layers"]["width"]
                state_2D = tf.concat(
                    axis=3, values=[features_2D, state_2D]
                )
                state_2D = self.layers.conv2D(
                    state_2D,
                    filters=2*channels_2D,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=self.layers.nonlinearity,
                    name="Conv1",
                    batchnorm=True,
                    mask=self.masks["conv2D"]
                ) * self.masks["conv2D"]
                state_2D = self.layers.normalize_layer(
                    state_2D, self.masks["conv2D"], name="2D"
                )
                state_2D = self.layers.conv2D_transpose(
                    state_2D,
                    filters=2*channels_2D,
                    kernel_size=[pair_width, pair_width],
                    padding="same",
                    activation=self.layers.nonlinearity,
                    name="Conv2"
                ) * self.masks["conv2D"]
                # Force field layers as 1x1 convolutions
                prod = self.layers.conv2D(state_2D,
                                        filters=num_pair_layers,
                                        kernel_size=[1, 1],
                                        padding="same",
                                        use_bias=True,
                                        name="Mix3")
                # Batch normalize
                field_layers = self.layers.normalize_layer(
                    prod, tf.expand_dims(self.masks["dists"], 3)
                )
                # Transpose 
                field_layers = tf.transpose(field_layers, [0, 3, 1, 2])

            # Each component of each field is sliced and processed
            # The fields are currently [Batch, layer, i, j]
            field_index = 0
            fields = {}
            force_loss = tf.zeros((1))

            # Softplus six-parameter potential
            with tf.variable_scope("ContactConv"):
                num_hidden = hyperparams["contact_conv"]["hidden"]
                width = hyperparams["contact_conv"]["width"]
                num_lengths = hyperparams["contact_conv"]["lengths"]
                num_visible = num_lengths + 3

                # Coefficients
                c0_unc = tf.transpose(
                    field_layers[:,field_index:field_index+num_visible,:,:],
                    [0,2,3,1]
                )
                field_index = field_index + num_visible
                c2_unc = tf.transpose(
                    field_layers[:,field_index:field_index+num_hidden,:,:],
                    [0,2,3,1]
                )
                field_index = field_index + num_hidden
                mask_expand = tf.expand_dims(self.masks["ij_long"], 3)
                c0 = _linear_interface(
                    c0_unc, "coeff_v", mask_expand, 0.0,
                    hyperparams["pair_scale_init"], positive=False
                )
                c2 = _linear_interface(
                    c2_unc, "coeff_contactconv", mask_expand, 1.0,
                    hyperparams["pair_scale_init"], positive=False
                )
                # Downsample
                c2 = tf.nn.avg_pool(
                    value=c2, 
                    ksize=[1, 1, 1, 1],
                    strides=[1, 4, 4, 1], 
                    padding="SAME"
                )

                _safe_summary("coeff2D", c2[:,:,:,:3])
                _safe_summary("coeff2D_v_RBF", c0[:,:,:,:3])
                _safe_summary("coeff2D_v_orient", c0[:,:,:,-3:])

                # Global parameters
                W1, b1 = _filter_bias("1", width, width, num_visible, num_hidden)
                W2, b2 = _filter_bias("2", width, width, num_hidden, num_hidden)

                D_loc_init = np.exp(np.linspace(np.log(2), np.log(20), num_lengths))
                D_loc_init = np.reshape(D_loc_init, [1,1,1,num_lengths])
                D_prec_init = 1. / np.square(D_loc_init)
                D_loc = tf.exp(tf.get_variable(
                    "D_loc", shape=[1,1,1,num_lengths], dtype=tf.float32,
                    initializer=tf.constant_initializer(np.log(D_loc_init)),
                    trainable=True
                ))
                D_prec = tf.exp(tf.get_variable(
                    "D_prec", shape=[1,1,1,num_lengths], dtype=tf.float32,
                    initializer=tf.constant_initializer(np.log(D_prec_init)),
                    trainable=True
                ))
                # Summary of the features
                W_D = W1[:,:,:3,:]
                _safe_summary("W1", tf.transpose(W_D, [3, 0, 1, 2]))
                fields["contact_conv"] = (W1, W2, b1, b2, c0, c2, D_loc, D_prec)
                fields["contact_conv_readonly"] = tuple(
                    tf.stop_gradient(x) for x in fields["contact_conv"]
                )

            with tf.variable_scope("SiteLayers"):
                num_site_layers = 13 + hyperparams["backbone_conv"]["lengths"] \
                    + self.dims["ss"] + hyperparams["backbone_conv"]["hidden"] * 2
                num_channels_1D = self.tensors["nodes"].get_shape().as_list()[3]
                up_1D = self.layers.convnet_1D(
                    self.tensors["nodes"], num_channels_1D, self.masks["conv1D"],
                    hyperparams["site_layers"]["conv1D"]["widths"],
                    hyperparams["site_layers"]["conv1D"]["dilations"],
                    self.placeholders["dropout"]
                )
                up_1D = self.layers.conv2D(
                    up_1D,
                    filters=2*channels_1D,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=self.layers.nonlinearity,
                    name="Mix1"
                ) * self.masks["conv1D"]
                up_1D = self.layers.conv2D_transpose(
                    up_1D,
                    filters=2*channels_1D,
                    kernel_size=[1, hyperparams["site_layers"]["width"]],
                    padding="same",
                    activation=self.layers.nonlinearity,
                    name="Mix2"
                ) * self.masks["conv1D"]
                up_1D = self.layers.normalize_layer(
                    up_1D, self.masks["conv1D"], name="A"
                )
                # RNN
                # Skip connecitons and postprocess
                up_1D = tf.concat(axis=3, values=[up_1D, self.tensors["nodes"]])
                up_1D = self.layers.conv2D(
                    up_1D,
                    filters=2*channels_1D,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=self.layers.nonlinearity,
                    name="MixFinal",
                    batchnorm=True,
                    mask=self.masks["conv1D"]
                ) * self.masks["conv1D"]
                # Force field layers as 1x1 convolutions
                up_layer = self.layers.conv2D(up_1D,
                                            filters=num_site_layers,
                                            kernel_size=[1, 1],
                                            padding="same",
                                            use_bias=True)
                up_layer = self.layers.normalize_layer(
                    up_layer, self.masks["conv1D"]
                )
                site_layers = tf.squeeze(up_layer, axis=1)
                site_layers = tf.transpose(site_layers, [0, 2, 1])

            # Each component of each field is sliced and processed
            # The fields are currently [Batch, layer, i]
            field_index = 0
            with tf.variable_scope("SS"):
                ss_pred_logits = _site_layer(
                    site_layers, field_index, channels=self.dims["ss"]
                )
                field_index += self.dims["ss"]
                ss_pred_logits = tf.unstack(ss_pred_logits, axis=1)
                ss_pred_logits = [
                    _linear_interface(
                        ss_pred_i, "ss_pred_" + str(i), self.masks["seqs"], 0,
                        hyperparams["site_scale_init"], positive=False
                    ) for i, ss_pred_i in enumerate(ss_pred_logits)
                ]
                self.tensors["ss_pre_logits"] = tf.stack(ss_pred_logits, axis=2)

            with tf.variable_scope("MassLayers"):
                num_mass_layers = 10
                # Post-process with reversed positional encodings
                idx = self.masks["indices"]
                lengths = tf.to_float(self.placeholders["lengths"])
                lengths_tiled = tf.tile(
                    tf.expand_dims(lengths, 1), [1, self.dims["length"]]
                )
                idx_flip = tf.abs(lengths_tiled - idx)
                scale = 100.
                mass_features = [
                    idx / scale, idx_flip / scale, lengths_tiled / scale,
                    tf.log(idx + 1.), tf.log(idx_flip + 1.), tf.log(lengths_tiled)
                ]
                mass_features = tf.stack(mass_features, axis=2)
                mass_features = tf.expand_dims(mass_features, 1)
                mass_features = self.layers.normalize_layer(
                    mass_features, self.masks["conv1D"]
                )
                mass_layers = tf.concat(
                    axis=3, values=[self.tensors["nodes"], mass_features]
                )
                num_channels_mass = mass_layers.get_shape().as_list()[3]
                mass_layers = self.layers.convnet_1D(
                    mass_layers, num_channels_mass, self.masks["conv1D"],
                    hyperparams["site_layers"]["conv1D"]["widths"],
                    hyperparams["site_layers"]["conv1D"]["dilations"],
                    self.placeholders["dropout"]
                )
                mass_layers = self.layers.conv2D(
                    mass_layers,
                    filters=num_mass_layers,
                    kernel_size=[1, 1],
                    padding="same",
                    activation=None,
                    name="Mix1",
                    batchnorm=True,
                    mask=self.masks["conv1D"]
                )
                mass_layers = tf.squeeze(mass_layers, axis=1)
                mass_layers = tf.transpose(mass_layers, [0, 2, 1])
            with tf.variable_scope("Backbone_lrA"):
                num_hidden = hyperparams["backbone_conv"]["hidden"]
                width = hyperparams["backbone_conv"]["width"]
                num_lengths = hyperparams["backbone_conv"]["lengths"]
                num_visible = 3 + num_lengths

                # Biases of RBM visible units
                b_v_unc = _site_layer(site_layers, field_index, channels=3 + num_lengths)
                # b_v_scales_unc = _site_layer(site_layers, field_index + 3 + num_lengths)
                field_index += 3 + num_lengths

                num_top_hidden = num_hidden * 2
                c3_unc = _site_layer(site_layers, field_index, channels=num_top_hidden)
                field_index += num_top_hidden
                c3_unc = tf.expand_dims(tf.transpose(c3_unc, [0,2,1]), 1)
                c3 = _linear_interface(
                    c3_unc, "coeff_backbone", self.masks["conv1D"], 1.0,
                    hyperparams["site_scale_init"], positive=False
                )
                # Downsample
                c3 = tf.nn.avg_pool(
                    value=c3, ksize=[1, 1, 1, 1], strides=[1, 1, 8, 1], padding="SAME"
                )
                _safe_summary("coeff", tf.transpose(c3, [0,3,2,1]))
                b_v_unc = tf.unstack(b_v_unc, axis=1)
                b_v_set = [
                    _linear_interface(
                        b_v_i_unc, "b_v_" + str(i), self.masks["seqs"], 0,
                        hyperparams["site_scale_init"], positive=False
                    ) for i, b_v_i_unc in enumerate(b_v_unc)
                ]
                b_v = tf.stack(b_v_set, axis=2)

                _safe_summary("b_v", tf.expand_dims(b_v[:,:,:3], 0))
                b_v = tf.expand_dims(b_v, 1)
                num_hidden_2D = hyperparams["contact_conv"]["hidden"]

                W1, b1 = _filter_bias("1", 1, 3, num_visible, num_hidden / 2)
                W2, b2 = _filter_bias("2", 1, 3, num_hidden / 2, num_hidden)
                W3, b3 = _filter_bias("3", 1, 3, num_hidden + num_hidden_2D, num_hidden * 2)

                L_init = np.ones([1,1,num_lengths])
                L_loc = tf.cumsum(tf.exp(tf.get_variable(
                    "L_loc", shape=[1,1,num_lengths], dtype=tf.float32,
                    initializer=tf.constant_initializer(L_init * np.log(5.)),
                    trainable=True
                )), 2)
                L_prec = tf.exp(tf.get_variable(
                    "L_prec", shape=[1,1,num_lengths], dtype=tf.float32,
                    initializer=tf.constant_initializer(L_init * np.log(0.1)),
                    trainable=True
                ))
                # Dropout on the hidden units
                h_mask = tf.nn.dropout(
                    tf.ones([self.dims["batch"], 1, self.dims["length"], num_hidden]), 
                    self.placeholders["dropout"]
                )
                # Summary of the features
                W_TP = W1[:,:,:3,:]
                _safe_summary("W_TP", tf.transpose(W_TP, [0, 3, 1, 2]))
                fields["backbone_conv"] = (W1,W2,W3,b1,b2,b3, c3, b_v, L_loc, L_prec, h_mask)
                fields["backbone_conv_readonly"] = tuple(
                    tf.stop_gradient(x) for x in fields["backbone_conv"]
                )
            with tf.variable_scope("Initialization_lrA"):
                fields["backbone"] = []
                # Predict Cartesian vectors for angles only
                # dX_init = _site_layer(site_layers, field_index, channels=3)
                dX_init_1_unc = _site_layer(site_layers, field_index)
                dX_init_2_unc = _site_layer(site_layers, field_index + 1)
                dX_init_3_unc = _site_layer(site_layers, field_index + 2)
                L_init_unc = _site_layer(site_layers, field_index + 3)
                field_index += 4
                # Parameterize lengths with an exponential for fast changes
                L_init = tf.expand_dims(_linear_interface(
                    L_init_unc, "length_init", self.masks["seqs"],
                    hyperparams["backbone"]["init"]["length"],
                    hyperparams["site_scale_init"],
                    positive=True
                ), 1)

                # Build initial bias
                bias_T = hyperparams["angles"]["init"]["angle"]
                bias_P = hyperparams["dihedrals"]["init"]["angle"]

                dX_init_1 = _linear_interface(
                    dX_init_1_unc, "dX_init1", self.masks["seqs"],
                    np.cos(bias_T),
                    hyperparams["site_scale_init"],
                    positive=False
                )
                dX_init_2 = _linear_interface(
                    dX_init_2_unc, "dX_init2", self.masks["seqs"],
                    np.sin(bias_T) * np.cos(bias_P),
                    hyperparams["site_scale_init"],
                    positive=False
                )
                dX_init_3 = _linear_interface(
                    dX_init_3_unc, "dX_init3", self.masks["seqs"],
                    np.sin(bias_T) * np.sin(bias_P),
                    hyperparams["site_scale_init"],
                    positive=False
                )
                dX_init = tf.stack([dX_init_1,dX_init_2,dX_init_3], axis=1)
                dX_init = L_init * _unit_vecs(dX_init)
                # Thermalize init in relative Cartesian coordinates
                dX_init += tf.random_normal(
                    [self.dims["batch"], 3, self.dims["length"]]
                )
                L_i, T_i, P_i = _cart_to_spherical(dX_init, self.masks["seqs"])
                fields["LTP_denovo"] = (L_i, T_i, P_i)

                # Initialize near the native state + noise
                def _coarse_slice(X, atom_ix=1):
                    offset = 3 * atom_ix
                    X = tf.slice(X, [0, 0, offset], [-1, -1, 3])
                    X = tf.transpose(X, [0, 2, 1])
                    return X
                # Mean-center all data
                X_data_fine = self.placeholders["coordinates_target"]
                mask = tf.expand_dims(self.masks["seqs"], 2)
                X_mean = tf.reduce_sum(mask * X_data_fine, 1, keep_dims=True) \
                    / tf.reduce_sum(mask, 1, keep_dims=True)
                X_data_fine = mask * (X_data_fine - X_mean) + (1- mask) * X_mean
                X_data = 0.5 * (
                    _coarse_slice(X_data_fine, atom_ix=4) 
                    + _coarse_slice(X_data_fine, atom_ix=1)
                )

                # Thermalize absolute Cartesian
                X_data +=  0.5 * tf.random_normal(
                    [self.dims["batch"], 3, self.dims["length"]]
                )

                # Repair loops with noisy smoothing
                smoothing_iterations = 10
                smoothing_width = 10
                X_data_expand = tf.expand_dims(tf.transpose(X_data, [0, 2, 1]), 1)
                mask_avg = tf.reshape(
                    self.placeholders["structure_mask"],
                    [self.dims["batch"], 1, self.dims["length"], 1]
                )
                for i in xrange(smoothing_iterations):
                    X_data_expand = mask_avg * X_data_expand \
                    + (1. - mask_avg) * (
                        tf.nn.avg_pool(
                            value=X_data_expand, 
                            ksize=[1, 1, smoothing_width, 1], 
                            strides=[1, 1, 1, 1], 
                            padding="SAME"
                        ) + 2. * tf.random_normal(tf.shape(X_data_expand))
                    )
                X_data = tf.transpose(tf.squeeze(X_data_expand,1), [0, 2, 1])

                # Interpolate with initializer
                L_data, T_data, P_data = self._dynamics_to_internal_coords(X_data)
                L_i, T_i, P_i = _cart_to_spherical(dX_init, self.masks["seqs"])
                # Fix N-terminus
                L_data = tf.concat(axis=1, values=[L_i[:,:3], L_data[:,3:]])
                T_data = tf.concat(axis=1, values=[T_i[:,:3], T_data[:,3:]])
                P_data = tf.concat(axis=1, values=[P_i[:,:3], P_data[:,3:]])
                # TODO: Remodel the loops
                dX_data = tf.stack(
                    [L_data * tf.cos(T_data), 
                     L_data * tf.sin(T_data) * tf.cos(P_data), 
                     L_data * tf.sin(T_data) * tf.sin(P_data)], 1
                )

                # Initialization loss - group sparsity version
                dX_dists_sq = tf.reduce_sum(tf.square(
                    tf.nn.l2_normalize(dX_data, 1) - tf.nn.l2_normalize(dX_init, 1)
                ), 1)
                L_dists = tf.square(tf.log(L_data + 1E-3) - tf.log(L_i + 1E-3))
                dX_dists = dX_dists_sq + L_dists
                dX_mask = self.placeholders["structure_mask"]
                pool_width = self.hyperparams["energy"]["backbone_conv"]["width"]
                dX_pooled = self.layers.pool_1D(dX_dists, dX_mask, pool_width)
                dX_pooled = tf.sqrt(dX_pooled + 1E-5)
                self.tensors["loss_init_dX"] = tf.reduce_sum(dX_mask * dX_pooled)\
                    / tf.reduce_sum(dX_mask)

                # Store whether or not to use native init
                u = tf.distributions.Bernoulli(
                    probs=self.placeholders["native_init_prob"] * tf.ones([self.dims["batch"], 1, 1]),
                    dtype=tf.float32
                ).sample()
                self.tensors["is_native"] = u
                dX_init = tf.cond(
                    self.placeholders["native_init_prob"] > 0.,
                    lambda: u * dX_data + (1. - u) * dX_init,
                    lambda: dX_init
                )

                # Convert unit vectors [B, 3, L] to lengths, angles, and dihedrals
                L_i, T_i, P_i = _cart_to_spherical(dX_init, self.masks["seqs"])
                
                # Summaries
                angle_image = tf.expand_dims(tf.stack(
                        [tf.cos(T_i), 
                         tf.sin(T_i) * tf.cos(P_i), 
                         tf.sin(T_i) * tf.sin(P_i)], 2
                    ) * tf.expand_dims(self.masks["seqs"], 2), 0)
                _safe_summary("angles_init", angle_image)

                # Initializations are used in NERF routines
                fields["lengths_init"] = L_i
                fields["angles_init"] = T_i
                fields["dihedrals_init"] = P_i

            with tf.variable_scope("Mass"):
                """ Cartesian mass matrix for HMC """
                # Diagonal of the precision matrix
                M_inv_sqrt_unc = _site_layer(mass_layers, 0, channels=3)

                # Linear interface for diagonals
                M_inv_init = hyperparams["mass"]["Z_inv"]
                M_inv_sqrt_unc = tf.unstack(M_inv_sqrt_unc, axis=1)
                M_inv_sqrt_set = [
                    _linear_interface(
                        M_inv_sqrt_i_unc, "M_inv_sqrt_" + str(i), self.masks["seqs"], 
                        np.sqrt(M_inv_init[i]), hyperparams["site_scale_init"],
                      positive=True
                    ) for i, M_inv_sqrt_i_unc in enumerate(M_inv_sqrt_unc)
                ]
                # (B,3,L)
                M_inv_sqrt = tf.stack(M_inv_sqrt_set, axis=1)
                M_inv = M_inv_sqrt * M_inv_sqrt
                
                # Summaries
                M_inv_img = tf.expand_dims(tf.transpose(M_inv,[1,0,2]), 3)
                _safe_summary("M_inv", M_inv_img)
                fields["mass"] = {}
                fields["mass"]["M_inv_sqrt"] = M_inv_sqrt
                fields["mass"]["M_inv"] = M_inv
        return fields, force_loss

    def _build_target(self):
        """ Build the coarse target coordinates and distance matrix """
        def _coarse_slice(X, atom_ix=1):
            offset = 3 * atom_ix
            X = tf.slice(X, [0, 0, offset], [-1, -1, 3])
            X = tf.transpose(X, [0, 2, 1])
            return X
        with tf.variable_scope("CoarseCoordinates"):
            X_data_fine = self.placeholders["coordinates_target"]

            # Mean-center all data
            mask = tf.expand_dims(self.masks["seqs"], 2)
            X_mean = tf.reduce_sum(mask * X_data_fine, 1, keep_dims=True) \
                / tf.reduce_sum(mask, 1, keep_dims=True)
            X_data_fine = mask * (X_data_fine - X_mean)

            if False:
                weights = self.tensors["energy"]["weights"]
                X_data_split = tf.reshape(
                    X_data_fine, [self.dims["batch"],
                                  self.dims["length"],
                                  self.dims["atoms"], 3]
                )
                # Weighted average over the atoms
                X_data_split = tf.expand_dims(weights, 3) * X_data_split
                X_data = tf.reduce_sum(X_data_split, axis=2)
                X_data = tf.transpose(X_data, [0, 2, 1])

            # Use the side chain center of mass as target
            X_data = 0.5 * (
                _coarse_slice(X_data_fine, atom_ix=4) 
                + _coarse_slice(X_data_fine, atom_ix=1)
            )
        with tf.variable_scope("Distance"):
            d_eps = self.hyperparams["scoring"]["logdists"]["eps"]
            # Determine contacts
            contact_cutoff_D = self.hyperparams["scoring"]["logdists"]["contact"]
            logD_data = 0.5 * tf.log(
                self._target_squared_dists(X_data) + d_eps
            )
            contacts_data = self.masks["structure_coarse_dists"] * tf.where(
                logD_data < np.log(contact_cutoff_D),
                tf.ones_like(logD_data), 
                tf.zeros_like(logD_data)
            )
        return X_data_fine, X_data, logD_data, contacts_data

    def _target_squared_dists(self, X):
        """Computes a batch of N distance matrices of size (N,L,L) from a batch
            of N coordinate sets (N,3,L)

            Use ||U-V|| = ||U|| + ||V|| - 2 U.V where ||U|| is the squared
            Euclidean norm
        """
        with tf.variable_scope("Distances"):
            norm = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
            D = norm + tf.transpose(norm, [0, 2, 1]) - 2 * \
                tf.matmul(X, X, adjoint_a=True)
        return D

    def _build_dynamics(self):
        """ Run multiple steps of Hamiltonian Monte Carlo """
        X_internal_init = self.tensors["coordinates_target"]

        with tf.variable_scope("Langevin"):
            # Learn the learning rate
            eps_init = self.hyperparams["folding"]["constants"]["rel_eps"]
            self.tensors["epsilon"] = tf.exp(tf.get_variable(
                "epsilon", (), 
                initializer=tf.constant_initializer(np.log(eps_init))
            ))
            tf.summary.scalar(
                "epsilon", self.tensors["epsilon"],
                collections=["gen_batch", "gen_dense"]
            )

            # Learn the temperature
            beta_init = self.hyperparams["folding"]["constants"]["beta"]
            self.tensors["beta"] = tf.exp(tf.get_variable(
                "beta", (), 
                initializer=tf.constant_initializer(np.log(beta_init))
            ))
            self.tensors["beta_readonly"] = tf.stop_gradient(self.tensors["beta"])
            tf.summary.scalar(
                "beta", self.tensors["beta"],
                collections=["gen_batch", "gen_dense"]
            )

            # Score denovo initializations
            L_i, T_i, P_i = self.tensors["energy"]["LTP_denovo"]
            X_init_denovo = self._dynamics_to_cartesian(L_i, T_i, P_i)
            init_score = 0. #self._dynamics_energy(X_init_denovo, stop_gradient=True)

            # Presample noise
            eMx = self.hyperparams["folding"]["constants"]["eMx"]
            eps = self.tensors["epsilon"]
            T = self.placeholders["langevin_steps"]
            self.tensors["noise_X"] = tf.sqrt(eMx) * tf.random_normal(
                [T, self.dims["batch"], 3, self.dims["length"]]
            )
            self.tensors["noise_Z"] = tf.sqrt(eps) * tf.random_normal(
                [T, self.dims["batch"], 3, self.dims["length"]]
            )

            # Sample retro
            retro_eps = 0.01
            retro_probs = 0.1 * tf.ones([T, self.dims["batch"]])
            zeroed_evens = tf.to_float(tf.floormod(tf.range(0, T, 1), 2))
            retro_probs = tf.expand_dims(zeroed_evens, 1) * retro_probs
            self.tensors["retro_steps"] = tf.distributions.Bernoulli(
                probs=retro_probs, dtype=tf.float32
            ).sample()
            self.tensors["retro_perturbations"] = retro_eps * \
                tf.random_normal(
                    [T, self.dims["batch"], 3, self.dims["length"]]
                )
            self.tensors["retro_perturbations"] = tf.reshape(
                self.tensors["retro_steps"], [T, self.dims["batch"], 1, 1]
            ) * self.tensors["retro_perturbations"]

            # Couple noise to preceding
            noise_X_shift = tf.pad(
                self.tensors["noise_X"][:-1,:,:,:], [[1,0],[0,0],[0,0],[0,0]]
            )
            noise_Z_shift = tf.pad(
                self.tensors["noise_Z"][:-1,:,:,:], [[1,0],[0,0],[0,0],[0,0]]
            )
            retro_shift = tf.pad(
                self.tensors["retro_steps"][:-1,:], [[1,0],[0,0]]
            )
            retro_shift = tf.reshape(retro_shift, [T, self.dims["batch"], 1, 1])
            self.tensors["noise_X"] = retro_shift * noise_X_shift \
                + (1. - retro_shift) * self.tensors["noise_X"]
            self.tensors["noise_Z"] = retro_shift * noise_Z_shift \
                + (1. - retro_shift) * self.tensors["noise_Z"]

            # Compute total steps
            num_total_steps = self.placeholders["langevin_steps"]
            # Collect state and kinetic energies
            X_set = tf.TensorArray(
                dtype=tf.float32, size=num_total_steps, dynamic_size=False
            )
            H_set = tf.TensorArray(
                dtype=tf.float32, size=num_total_steps, dynamic_size=False
            )
            dXdZ_init = tf.zeros([self.dims["batch"],
                                  self.dims["length"],
                                  self.dims["length"],
                                  3])
            X_init, self.tensors["loss_init_RG"] = self._dynamics_build_initial_coords()
            U_init = tf.zeros([self.dims["batch"]])
            zero = tf.constant(0, dtype="int32")
            zero_float = tf.constant(0., dtype="float32")
            state_init = [zero, zero_float, X_init, U_init, H_set, X_set]
            loop_out = tf.while_loop(
                lambda i, loss, x, u, H, X: i < num_total_steps,
                self._dynamics_langevin_step,
                state_init, swap_memory=True)
            _, dynamics_loss, X_final, _, H_set, X_set_final = loop_out
            
            # Repack TensorArrays
            # State trajectory
            X_trajectory = tf.reshape(
                X_set_final.concat(),
                [num_total_steps, self.dims["batch"], 3, self.dims["length"]]
            )
            X_trajectory = tf.transpose(X_trajectory, [1, 0, 2, 3])
            # Kinetic energy trajectory
            H_set = tf.reshape(
                H_set.concat(), [num_total_steps, 2, self.dims["batch"]]
            )
            H_set = tf.transpose(H_set, [2, 0, 1])

            # Acceptance probabilities
            X_logprob = -tf.nn.relu(H_set[:,1] - H_set[:,0])

            # Scale dynamics_loss by batch size and tracectory
            dynamics_loss /= tf.to_float(num_total_steps)
        return X_final, X_trajectory, X_logprob, init_score, dynamics_loss

    def _dynamics_build_initial_coords(self):
        """ Instantiate unfolded coordinates """
        # Bond lengths
        fields = self.tensors["energy"]
        L_init = fields["lengths_init"]
        T_init = fields["angles_init"]
        P_init = fields["dihedrals_init"]
        X = self._dynamics_to_cartesian(L_init, T_init, P_init)
        with tf.variable_scope("Score"):
            # Compute radius of gyration in each protein
            d_eps = self.hyperparams["folding"]["constants"]["dist_eps"]
            X_com = tf.reduce_sum(
                self.masks["forces"] * X, axis=2, keep_dims=True
            ) / tf.reduce_sum(self.masks["forces"], axis=2, keep_dims=True)
            D_sq = tf.reduce_sum(tf.square(X - X_com), axis=[1])
            mean_Dsq = tf.reduce_sum(self.masks["seqs"] * D_sq, axis=[1]) \
                / tf.reduce_sum(self.masks["seqs"], axis=[1])
            log_RG = 0.5 * tf.log(mean_Dsq + d_eps)
            # Null radius of gyration
            lengths = tf.to_float(self.placeholders["lengths"])
            R0 = self.hyperparams["folding"]["init_score"]["R0"]
            v = self.hyperparams["folding"]["init_score"]["v"]
            coeff = self.hyperparams["folding"]["init_score"]["coeff"]
            log_null_RG = np.log(R0) + v * tf.log(lengths)
            scores = coeff * tf.nn.relu(log_null_RG - log_RG)
            # Only score non-native initializations
            scores = (1 - self.tensors["is_native"]) * scores
            score = tf.reduce_mean(scores)
        with tf.variable_scope("Lengths"):
            # Determine average bond length for each protein
            dXi = X[:,:,1:] - X[:,:,:-1]
            Li = tf.sqrt(tf.reduce_sum(tf.square(dXi), 1) + d_eps)
            mask = self.masks["backbone"]
            avL_init = tf.reduce_sum(mask * Li, 1) / tf.reduce_sum(mask, 1)
            self.tensors["coarse_avL_init"] = avL_init
        return X, score

    def _dynamics_to_cartesian(self, length, theta, phi):
        """ Converts internal coordinates Z to Cartesian coords X """
        def _normed(X):
            X_norm = tf.sqrt(tf.reduce_sum(X * X, 1, keep_dims=True))
            return X / X_norm

        def _input_ta(X=None):
            ta = tf.TensorArray(dtype=tf.float32, size=self.dims["length"],
                                dynamic_size=False,
                                element_shape=[None, 1])
            # Length first
            X = tf.transpose(X)
            ta = ta.unstack(tf.expand_dims(X, 2))
            return ta

        def _add_next_bead(i, u2, u1, x_i, L_ta, CT_ta, CPST_ta, SPST_ta,
                           X_ta):
            """ Extend the chain with SNERF """
            # Read the current timestep data, (batch_size, 3)
            length_i = L_ta.read(i)
            cos_theta_i = CT_ta.read(i)
            cos_phi_sin_theta_i = CPST_ta.read(i)
            sin_phi_sin_theta_i = SPST_ta.read(i)
            # Compute normals and next direction vector
            n_a = _normed(tf.cross(u2, u1))
            n_b = tf.cross(n_a, u1)
            u_new = cos_theta_i * u1 + cos_phi_sin_theta_i * n_b \
                + sin_phi_sin_theta_i * n_a
            # Offset at length_i
            x_i = x_i + length_i * u_new
            #
            X_ta = X_ta.write(i, x_i)
            return i+1, u1, u_new, x_i, L_ta, CT_ta, CPST_ta, SPST_ta, X_ta

        with tf.variable_scope("NERF"):
            # Initialize tensor array for Cartesian coordinates
            X_ta = tf.TensorArray(dtype=tf.float32, size=self.dims["length"],
                                  dynamic_size=False)
            # Precompute trigonemtric ratios
            theta_complement = np.pi - theta
            cos_theta = tf.cos(theta_complement)
            sin_theta = tf.sin(theta_complement)
            cos_phi_sin_theta = tf.cos(phi) * sin_theta
            sin_phi_sin_theta = tf.sin(phi) * sin_theta
            # Stage all precomputed values into TensorArrays
            L_ta = _input_ta(length)
            CT_ta = _input_ta(cos_theta)
            CPST_ta = _input_ta(cos_phi_sin_theta)
            SPST_ta = _input_ta(sin_phi_sin_theta)
            # Initialization
            zero = tf.constant(0, dtype=tf.int32)
            # Seed coordinates
            z_block = tf.zeros([self.dims["batch"], 3])
            x_i3_init = z_block + tf.constant([[0., 0., 0.]])
            x_i2_init = z_block + tf.constant([[1., 0., 0.]])
            x_i1_init = z_block + tf.constant([[1., 1., 1.]])
            u2_init = _normed(x_i2_init - x_i3_init)
            u1_init = _normed(x_i1_init - x_i2_init)
            state_init = [zero, u2_init, u1_init, x_i1_init,
                          L_ta, CT_ta, CPST_ta, SPST_ta, X_ta]
            # Iteratively build the coordinates with NERF
            _, _, _, _, _, _, _, _, X_ta = tf.while_loop(
                lambda i, x_i3, x_i2, x_i1, L, CT, CPST, SPST, X:
                i < self.dims["length"],
                _add_next_bead,
                state_init)
            # Flatten back into a tensor
            X = tf.reshape(X_ta.concat(),
                           [self.dims["length"], self.dims["batch"], 3])
            X = tf.transpose(X, [1, 2, 0])
        return X

    def _dynamics_to_internal_coords(self, X):
        constants = self.hyperparams["folding"]["constants"]
        _dynamics_eps = constants["xz_eps"]
        def _normed_cross(A, B):
            cross = tf.cross(A, B)
            mag = tf.sqrt(
                tf.reduce_sum(tf.square(cross), 2,
                              keep_dims=True) + _dynamics_eps
            )
            return cross / mag
        with tf.variable_scope("XtoZ"):
            with tf.variable_scope("Backbone"):
                # Need three offsets of X (With NERF init coords)
                X_init_coords = np.asarray(
                    [[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]]).T
                X_init_coords = tf.to_float(X_init_coords)
                z_block = tf.zeros([self.dims["batch"], 3, 3])
                X_init_tile = z_block + tf.expand_dims(X_init_coords, 0)
                # (B,3,L)
                X = tf.concat(axis=2, values=[X_init_tile, X])
                # L + 2 difference vectors
                dXi_shape = [self.dims["batch"], 3, self.dims["length"]+2]
                dXi = tf.slice(X, [0, 0, 1], dXi_shape) \
                    - tf.slice(X, [0, 0, 0], dXi_shape)
                Di = tf.sqrt(tf.reduce_sum(
                    tf.square(dXi) + _dynamics_eps, 1))
                dXi_unit = dXi / tf.expand_dims(Di, 1)
                # (B,3,L+2) => (B,L+2,3) => 3x (B,L,3)
                u_trans = tf.transpose(dXi_unit, [0, 2, 1])
                z_shape = [self.dims["batch"], self.dims["length"], 3]
                u_minus_2 = tf.slice(u_trans, [0, 0, 0], z_shape)
                u_minus_1 = tf.slice(u_trans, [0, 1, 0], z_shape)
                u_minus_0 = tf.slice(u_trans, [0, 2, 0], z_shape)
            with tf.variable_scope("Lengths"):
                length = tf.slice(
                    Di, [0, 2], [self.dims["batch"], self.dims["length"]])
            with tf.variable_scope("Angles"):
                theta = tf.acos(-tf.reduce_sum(u_minus_1 * u_minus_0, 2))
            with tf.variable_scope("Dihedrals"):
                norms_minus_2 = _normed_cross(u_minus_2, u_minus_1)
                norms_minus_1 = _normed_cross(u_minus_1, u_minus_0)
                phi = tf.sign(tf.reduce_sum(u_minus_2 * norms_minus_1, 2)) \
                    * tf.acos(tf.reduce_sum(norms_minus_2 * norms_minus_1, 2))
        return length, theta, phi

    def _dynamics_jacobian(self, X):
        constants = self.hyperparams["folding"]["constants"]
        J_eps = constants["jacob_eps"]
        with tf.variable_scope("Jacobian"):
            # Need three offsets of X (With NERF init coords)
            X_init = np.tile(np.asarray([[1, 0, 0], [1, 1, 1]]).T, (1, 1, 1))
            X_init_tile = X_init + tf.zeros([self.dims["batch"], 3, 2])
            X_full = tf.concat(axis=2, values=[X_init_tile, X])

            # Unit backbone vectors
            # (B,3,L) => (B,L,3) for ops on XYZ
            X_full_trans = tf.transpose(X_full, [0, 2, 1])
            shape_plus_1 = [self.dims["batch"], self.dims["length"]+1, 3]
            X_trans_shift_0 = tf.slice(X_full_trans, [0, 1, 0], shape_plus_1)
            X_trans_shift_1 = tf.slice(X_full_trans, [0, 0, 0], shape_plus_1)
            # Difference vectors
            dX_expand = X_trans_shift_0 - X_trans_shift_1
            mag_expand = tf.sqrt(
                tf.reduce_sum(tf.square(dX_expand) + J_eps, 2, keep_dims=True)
            )
            U_expand = dX_expand / mag_expand
            shape_plus_0 = [self.dims["batch"], self.dims["length"], 3]
            Ui_s_0 = tf.slice(U_expand, [0, 1, 0], shape_plus_0)
            Ui_s_1 = tf.slice(U_expand, [0, 0, 0], shape_plus_0)

            # Angle normals (axes of rotation)
            T_normals = tf.cross(Ui_s_0, Ui_s_1)
            T_normals_mag = tf.sqrt(
                tf.reduce_sum(tf.square(T_normals), 2, keep_dims=True) + J_eps)
            T_normals = T_normals / T_normals_mag

            # Radial vectors X_j - X_i-1
            shape_plus_0_trans = [self.dims["batch"], 3, self.dims["length"]]
            X_shift_0 = tf.slice(X_full, [0, 0, 2], shape_plus_0_trans)
            X_shift_1 = tf.slice(X_full, [0, 0, 1], shape_plus_0_trans)
            Rij = tf.expand_dims(X_shift_0, 2) - tf.expand_dims(X_shift_1, 3)
            # (B,3,L,L) => (B,L,L,3)
            Rij = tf.transpose(Rij, [0, 2, 3, 1])

            # Tile the relevant vectors over j
            # No current support for broadcasting cross products
            tile_vec = [1, 1, self.dims["length"], 1]
            Ui_s_0_tile = tf.tile(tf.expand_dims(Ui_s_0, 2), tile_vec)
            Ui_s_1_tile = tf.tile(tf.expand_dims(Ui_s_1, 2), tile_vec)
            T_normals_tile = tf.tile(tf.expand_dims(T_normals, 2), tile_vec)

            # The Jacobian elements are each (B,L,L,3)
            # (Batch, z_i, x_j, xyz)
            dXdL = self.masks["jacobian"] * Ui_s_0_tile
            dXdT = self.masks["jacobian"] * tf.cross(T_normals_tile, Rij)
            dXdP = self.masks["jacobian"] * tf.cross(Ui_s_1_tile, Rij)
        return dXdL, dXdT, dXdP

    def _dynamics_forces_Z(self, X, t, dUdX, stop_gradient=False):
        # Initialize tensor array for internal coordinates
        constants = self.hyperparams["folding"]["constants"]
        fields = self.tensors["energy"]
        d_eps = constants["dist_eps"]
        with tf.variable_scope("Forces"):
            dXdL, dXdT, dXdP = self._dynamics_jacobian(X)
            with tf.variable_scope("ApplyJacobian"):
                # Apply Jacobian
                # (B,3,L) => (B,1,L,3)
                dUdX_expand = tf.expand_dims(tf.transpose(dUdX, [0, 2, 1]), 1)
                dUdL = tf.reduce_sum(dUdX_expand * dXdL, [2, 3])
                dUdT = tf.reduce_sum(dUdX_expand * dXdT, [2, 3])
                dUdP = tf.reduce_sum(dUdX_expand * dXdP, [2, 3])
            with tf.variable_scope("Unconstrain"):
                L, T, P = self._dynamics_to_internal_coords(X)
                # Convert derivatives to unconstrained coordinates
                dLdL_unc =  (1 - tf.exp(-L))
                dTdT_unc = T * (1 - (T / np.pi))
                dUdL_unc = dUdL * dLdL_unc
                dUdT_unc = dUdT * dTdT_unc
            with tf.variable_scope("Restack"):
                dUdZ = tf.stack([dUdL_unc, dUdT_unc, dUdP])
                # (3,B,L) => (B,3,L)
                dUdZ = tf.transpose(dUdZ, [1, 0, 2])
            with tf.variable_scope("Loss"):
                loss = 0
        return dUdZ, dXdL, dXdT, dXdP, L, T, P, loss

    def _dynamics_forces_X(self, X, t, stop_gradient=False):
        # Initialize tensor array for internal coordinates
        constants = self.hyperparams["folding"]["constants"]
        fields = self.tensors["energy"]
        d_eps = constants["dist_eps"]
        with tf.variable_scope("Forces"):
            with tf.variable_scope("BackboneFeatures"):
                L, T, P = self._dynamics_to_internal_coords(X)
                # Restricted Boltzmann Machine governs backbone angles
                key = "backbone_conv_readonly" if stop_gradient else "backbone_conv"
                # W, b_h, b_v, L_loc, L_prec, h_mask = fields[key]
                W1, W2, W3, b1, b2, b3, c3, b_v, L_loc, L_prec, h_mask = fields[key]
                # Unit vectors encode angles, RBF distances
                v_TP = [tf.cos(T), tf.sin(T) * tf.cos(P), tf.sin(T) * tf.sin(P)]
                v_TP = tf.stack(v_TP, 2)
                v_L = tf.exp(-L_prec * tf.square(tf.expand_dims(L,2) - L_loc))
                v_1D = tf.expand_dims(tf.concat(
                    axis=2, values = [v_TP, v_L]
                ), 1) * self.masks["conv1D"]
            with tf.variable_scope("Distances"):
                # Compute pairwise residuals (N,3,L,L)
                dX = tf.expand_dims(X, 3) - tf.expand_dims(X, 2)
                D = tf.sqrt(tf.reduce_sum(tf.square(dX), 1) + d_eps)
            with tf.variable_scope("Orientation"):
                # Build local reference frames
                L2_eps = 1E-2
                X_trans = tf.transpose(X, [0,2,1])
                u_i = tf.nn.l2_normalize(X_trans[:,1:,:] - X_trans[:,:-1,:], 2, epsilon=L2_eps)
                u_bw = u_i[:,:-1,:]
                u_fw = u_i[:,1:,:]
                e_i = tf.nn.l2_normalize(u_bw - u_fw, 2, epsilon=L2_eps)
                a_i = tf.nn.l2_normalize(tf.cross(u_bw, u_fw), 2, epsilon=L2_eps)
                exa_i = tf.cross(e_i, a_i)
                R = tf.stack([e_i, a_i, exa_i], 2)
                R = tf.pad(R, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")
                # Unit difference vectors
                # (N, L, 1, 3, 3) * (N, L, L, 1, 3)
                dX_trans = tf.transpose(dX, [0,2,3,1])
                u_ij = tf.nn.l2_normalize(dX_trans, 3, epsilon=L2_eps)
                # Rotate difference vectors into local coordinate systems
                # This expansion/reduction was faster than the tf.einsum macro
                v_R = tf.reduce_sum(tf.expand_dims(R, 2) * tf.expand_dims(u_ij, 3), 4)
            with tf.variable_scope("ContactConv"):
                key = "contact_conv_readonly" if stop_gradient else "contact_conv"
                W1, W2, b1, b2, c0, c2, D_loc, D_prec = fields[key]
                mask_expand = tf.expand_dims(self.masks["ij_long"], 3)
                # 2D featurization
                v_D = tf.exp(-D_prec * tf.square(tf.expand_dims(D,3) - D_loc))
                v_D = mask_expand * tf.concat(axis=3, values=[v_D, v_R])
                h1_2D = tf.nn.softplus(tf.nn.convolution(
                    v_D, filter=W1, strides=(2, 2), padding="SAME"
                ) + b1)
                h2_2D = tf.nn.softplus(tf.nn.convolution(
                    h1_2D, filter=W2, strides=(2, 2), padding="SAME"
                ) + b2) * self.masks["ij_long_x4"]
                Uij_conv = -tf.reduce_sum(c2 * h2_2D) - tf.reduce_sum(c0 * v_D)
            with tf.variable_scope("BackboneConv"):
                # Backbone convolutional net
                key = "backbone_conv_readonly" if stop_gradient else "backbone_conv"
                # W, b_h, b_v, L_loc, L_prec, h_mask = fields[key]
                W1, W2, W3, b1, b2, b3, c3, b_v, L_loc, L_prec, h_mask = fields[key]
                h1_1D = tf.nn.softplus(tf.nn.convolution(
                    v_1D, filter=W1, strides=(2, 2), padding="SAME"
                ) + b1)
                h2_1D = tf.nn.softplus(tf.nn.convolution(
                    h1_1D, filter=W2, strides=(2, 2), padding="SAME"
                ) + b2)
                # Aggregate 2D features for 1D
                v_h2_2D = tf.reduce_sum(h2_2D, 1, keep_dims=True)
                v_h2_2D = tf.nn.l2_normalize(v_h2_2D, 3, epsilon=1E-3)
                h2_1D = tf.concat(axis=3,values=[h2_1D, v_h2_2D])
                h3_1D = c3 * tf.nn.softplus(tf.nn.convolution(
                    h2_1D, filter=W3, strides=(2, 2), padding="SAME"
                ) + b3)
                Ui_reduce = -tf.reduce_sum(h3_1D, [1,2,3]) \
                     -tf.reduce_sum(b_v * v_1D, [1,2,3])
                Uz = tf.reduce_sum(Ui_reduce)
                dUzdX = tf.gradients(Uz, X)[0]
            with tf.variable_scope("NetForces"):
                with tf.variable_scope("Pairwise"):
                    dUxdX = tf.gradients(Uij_conv, X)[0]
            with tf.variable_scope("Energies"):
                U = Ui_reduce
                U *= self.tensors["beta"]
                dUxdX *= self.tensors["beta"]
                dUzdX *= self.tensors["beta"]
                dUdX = dUxdX + dUzdX
            with tf.variable_scope("Loss"):
                force_sq = tf.reduce_sum(
                    self.masks["seqs"] * tf.reduce_sum(
                        tf.square(dUdX), 1
                    ), 1
                )
                force_sq /= tf.reduce_sum(self.masks["seqs"], 1)
                force_rms = tf.sqrt(force_sq + 1E-5)
                loss = tf.reduce_mean(force_rms)
        return dUdX, dUxdX, L, T, P, U, loss

    def _dynamics_sample_noise(self):
        """ Returns resampled momentum variables for HMC """
        with tf.variable_scope("MomentumSample"):
            Z = tf.random_normal([self.dims["batch"], 3, self.dims["length"]])
            V = self.masks["forces"] * M_inv_sqrt * Z
        return V

    def _dynamics_flatten_coords(self, P):
        """ Convert (B,3,L) coords to (B,L3,1) """
        with tf.variable_scope("Flatten"):
            # (B,3,L) => (B,L,3)
            P_trans = tf.transpose(P, [0,2,1])
            # (B,L,3) => (B,L3,1)
            P_flat = tf.reshape(P_trans, [self.dims["batch"], self.dims["length"] * 3, 1])
        return P_flat

    def _dynamics_unflatten_coords(self, P_flat):
        """ Convert (B,L3,1) coords to (B,3,L) """
        with tf.variable_scope("Unflatten"):
            P_trans = tf.reshape(P_flat, [self.dims["batch"], self.dims["length"], 3])
            P = tf.transpose(P_trans, [0,2,1])
        return P

    def _dynamics_multiply_mass(self, P):
        """ Rescale momenta by inverse Mass matrix """
        with tf.variable_scope("MassScale"):
            # Mass is (B,L3,L3)
            M_inv = self.tensors["energy"]["mass"]["M_inv"]
            scaled_P = M_inv * P
        return scaled_P

    def _dynamics_multiply_jacobian(self, dZ, dXdL, dXdT, dXdP, 
        L=None, T=None, X=None):
        """ Transform dZ to dZ by multiplying by Jacobian dXdZ """
        # Each Jacobian is (B,L,L,3)
        # Broadcast dZ (B,3,L) => 3x(B,L)
        dL_unc, dT_unc, dP = tf.unstack(dZ, axis=1)
        if L is None or T is None:
            L, T, P =self._dynamics_to_internal_coords(X)
        dL = (1 - tf.exp(-L)) * dL_unc
        dT = T * (1 - (T / np.pi)) * dT_unc
        dZ_shape = [self.dims["batch"], self.dims["length"], 1, 1]
        dL = self.masks["dL"] * tf.reshape(dL, dZ_shape)
        dT = self.masks["dT"] * tf.reshape(dT, dZ_shape)
        dP = self.masks["dP"] * tf.reshape(dP, dZ_shape)
        dX = tf.transpose(
            tf.reduce_sum(dXdL * dL, 1)
            + tf.reduce_sum(dXdT * dT, 1)
            + tf.reduce_sum(dXdP * dP, 1),
            [0,2,1]
        )
        return dX

    def _dynamics_langevin_step(self, i, loss, X_prev, U, H_set, X_set):
        """ Integrates a leapfrog step for N independent systems with positions 
            X (N,3,L), momenta V (N,3,L), and forces F (N,3,L) """
        langevin_steps = self.placeholders["langevin_steps"]
        constants = self.hyperparams["folding"]["constants"]
        eps = self.tensors["epsilon"]

        with tf.variable_scope("Langevin"):
            # Fractional coordinate within simulation-
            i_frac = (tf.to_float(i) + 1.) / tf.to_float(langevin_steps)
            i_next = i + 1
            delta_loss = 0

            # Total energy after resampling
            H_init = tf.zeros_like(U)

            # Cartesian Langevin step
            X_0 = X_prev + self.tensors["retro_perturbations"][i,:,:,:]
            dUdX, dUxdX, _, _, _, _, loss_force_X = self._dynamics_forces_X(X_0, i_frac)
            eMx = constants["eMx"]
            dX_cart = -0.5 * eMx * dUdX + self.tensors["noise_X"][i,:,:,:]
            X_1 = X_0 + dX_cart

            # Force calculation
            dUdZ, dXdL, dXdT, dXdP, L, T, _, loss_force_Z = \
                self._dynamics_forces_Z(X_1, i_frac, dUdX)
            M_dUdZ = self._dynamics_multiply_mass(dUdZ)
            # Multiply noise by mass matrix
            noise = self.tensors["noise_Z"][i,:,:,:]
            M_inv_sqrt = self.tensors["energy"]["mass"]["M_inv_sqrt"]
            V = self.masks["forces"] * M_inv_sqrt * noise
            dZ = -0.5 * eps * M_dUdZ + V
            dX_pred = self._dynamics_multiply_jacobian(dZ, dXdL, dXdT, dXdP, L, T)

            with tf.variable_scope("TimeScale"):
                # Time rescaling
                X_center = tf.transpose(X_1,[0,2,1])
                dX = tf.transpose(dX_pred,[0,2,1])
                mask = tf.expand_dims(self.masks["seqs"], 2)
                denom = tf.reduce_sum(mask, 1, keep_dims=True)
                # Translational component
                dX_translational = tf.reduce_sum(dX * mask, 1, keep_dims=True) / denom
                dX_notrans = dX - dX_translational
                # Rotational component
                X_center = X_center - (tf.reduce_sum(X_center * mask, 1, keep_dims=True)) / denom
                L_rot = tf.cross(X_center, dX_notrans)
                L_total = tf.reduce_sum(L_rot * mask, 1, keep_dims=True)
                omega = L_total / tf.reduce_sum(tf.square(X_center) * mask, [1,2], keep_dims=True)
                omega = omega * tf.ones_like(X_center)
                # Approximate rotation by numerical integration
                dX_1 = tf.cross(omega, X_center)
                dX_2 = tf.cross(omega, X_center + dX_1)
                dX_angular = 0.5 * (dX_1 + dX_2)
                dX_detrended = dX_notrans - dX_angular
                # v_0_sq = 0.5
                # v_0_sq = 1.0
                # speed_max_sq = 1.0
                speed_weight = constants["speed_cost"]
                speed_max = 2.0
                speed_max_sq = speed_max * speed_max
                # K = tf.reduce_sum(sq_dX, 1)
                # K_0 = tf.reduce_sum(self.masks["seqs"] * v_0_sq, 1)
                # rescale = tf.sqrt(K_0 / K)
                # Compute max speed clipping
                sq_dX = self.masks["seqs"] * tf.reduce_sum(tf.square(dX_detrended), 2)
                max_sq = tf.reduce_max(sq_dX, 1)
                rescale = tf.minimum(1., tf.sqrt(speed_max_sq / max_sq))
                rescale = tf.reshape(rescale, [self.dims["batch"], 1, 1])
                # Regularize overflow
                norm_dX = tf.sqrt(sq_dX + 1.E-3)
                speed_penalty = tf.nn.softplus(speed_weight * (norm_dX - speed_max))
                delta_loss += tf.reduce_sum(self.masks["seqs"] * speed_penalty) \
                    / tf.reduce_sum(self.masks["seqs"])

            with tf.variable_scope("Corrector"):
                dX_pred = rescale * dX_pred 
                X_pred = X_1 + dX_pred
                # Corrector step
                dXdL_pred, dXdT_pred, dXdP_pred = self._dynamics_jacobian(X_pred)
                dX_corr = self._dynamics_multiply_jacobian(
                    rescale * dZ, dXdL_pred, dXdT_pred, dXdP_pred, X=X_pred
                )
                dX_int = 0.5 * (dX_pred + dX_corr)
                X_2 = X_1 + dX_int

            # ========================== Cartesian coordinate ===========================
            loss_force = loss_force_X + loss_force_Z

            # Rotate onto previous state
            with tf.variable_scope("Rotate"):
                dX = X_2 - X_0
                # Remove net translational and rotational components of velocity
                X_center = tf.transpose(X_0,[0,2,1])
                dX = tf.transpose(dX,[0,2,1])
                mask = tf.expand_dims(self.masks["seqs"], 2)
                denom = tf.reduce_sum(mask, 1, keep_dims=True)
                # Translational component
                dX_translational = tf.reduce_sum(dX * mask, 1, keep_dims=True) / denom
                dX_notrans = dX - dX_translational
                # Rotational component
                X_center = X_center - (tf.reduce_sum(X_center * mask, 1, keep_dims=True)) / denom
                L_rot = tf.cross(X_center, dX_notrans)
                L_total = tf.reduce_sum(L_rot * mask, 1, keep_dims=True)
                omega = L_total / tf.reduce_sum(tf.square(X_center) * mask, [1,2], keep_dims=True)
                omega = omega * tf.ones_like(X_center)
                # Approximate rotation by numerical integration
                dX_1 = tf.cross(omega, X_center)
                dX_2 = tf.cross(omega, X_center + dX_1)
                dX_angular = 0.5 * (dX_1 + dX_2)
                dX_detrended = dX_notrans - dX_angular
                X = X_0 + tf.transpose(dX_detrended, [0,2,1])

            delta_loss += constants["force_norm_coeff"] * loss_force
            loss = loss + delta_loss

            # Gradient damping
            decay = tf.exp(self.placeholders["bptt_log_decay"])
            X = decay * X + (1. - decay) * tf.stop_gradient(X)

            # Write the trajectory
            X_set = X_set.write(i, X)

        # Kinetic energy before and after resampling
        H_final = tf.zeros_like(U)
        H_stack = tf.stack([H_init, H_final])
        H_set = H_set.write(i, H_stack)

        # Retrospective replacement for sensitivity analysis
        retro_mask = tf.reshape(
            self.tensors["retro_steps"][i,:], [self.dims["batch"], 1, 1]
        ) * tf.ones_like(X)
        X = tf.where(retro_mask > 0, X_prev, X)

        # Update counts
        return i_next, loss, X, U, H_set, X_set

    def _build_atomizer(self):
        """Builds the graph for the atomic reconstruction"""

        constants = self.hyperparams["folding"]["constants"]
        _dynamics_eps = constants["xz_eps"]
        lengths = self.placeholders["lengths"]
        X = self.tensors["coordinates_coarse"]
        seqs = self.tensors["seqs_1D"]
        mask = self.masks["conv1D"]

        # RDConv params
        hyperparams = self.hyperparams["reconstruction"]
        widths = hyperparams["conv_widths"]
        dilations = hyperparams["conv_dilations"]

        # Input layers
        # num_channels_1D = self.tensors["nodes"].get_shape().as_list()[3]
        # self.tensors["nodes" = features_1D (seqs_1D, cnn_1D), state_1D
        num_channels_1D = self.hyperparams["energy"]["graph_conv"]["channels_1D"] \
            + self.dims["alphabet"] \
            + self.hyperparams["energy"]["cnn"]["out_channels"]

        # Output channels
        num_vectors = self.dims["atoms"] + 1
        num_out = self.dims["ss"] + num_vectors * 4

        def _normed_cross(A, B):
            cross = tf.cross(A, B)
            mag = tf.sqrt(
                tf.reduce_sum(tf.square(cross), 2, keep_dims=True)
                + _dynamics_eps
            )
            return cross / mag

        with tf.variable_scope("Atomizer"):
            reconstruction_loss = tf.zeros((1))

            # Encode log bond lenths and Sin and Cos of angles
            L, T, P = self._dynamics_to_internal_coords(X)
            L, T, P = [tf.expand_dims(x, 2) for x in [L, T, P]]
            # Latent features
            in_features = [
                tf.log(L), tf.cos(T), tf.sin(T)*tf.cos(P),
                tf.sin(T)*tf.sin(P), tf.squeeze(seqs, 1)
            ]
            in_features = tf.concat(axis=2, values=in_features)
            # Preprocess
            in_features = tf.expand_dims(in_features, 1)
            in_features  = self.layers.conv2D(
                in_features, filters=num_channels_1D,
                kernel_size=[1, 1], padding="same",
                activation=self.layers.nonlinearity, name="Mix1",
                batchnorm=True,
                mask=self.masks["conv1D"]
            ) * self.masks["conv1D"]
            # ConvNet followed by linear transform
            out_layer = self.layers.convnet_1D(
                in_features, num_channels_1D, 
                mask, widths, dilations,
                self.placeholders["dropout"]
            )
            # Dense prediction
            out_layer = self.layers.normalize_layer(
                out_layer, self.masks["conv1D"], 
                name="A"
            )
            out_layer = self.layers.conv2D(
                out_layer, filters=2*num_channels_1D,
                kernel_size=[1, 1], padding="same",
                activation=self.layers.nonlinearity, name="Mix2"
            )
            out_layer = self.layers.normalize_layer(
                out_layer, self.masks["conv1D"], name="B"
            )
            out_layer = self.layers.conv2D(
                out_layer, filters=num_out,
                kernel_size=[1, 1], padding="same",
                activation=None, name="MixOut"
            )
            out_layer = tf.squeeze(out_layer, 1)
            SS = out_layer[:, :, :self.dims["ss"]]
            dX = out_layer[:, :, self.dims["ss"]:self.dims["ss"]+num_vectors*3]
            L_unc = out_layer[:, :, self.dims["ss"]+num_vectors*3:self.dims["ss"]+num_vectors*4]

            with tf.variable_scope("NERF"):
                # Build local reference frames
                # Extend the ends of X to be coplanar
                X_pre = tf.expand_dims(X[:, :, 0] + X[:, :, 1] - X[:, :, 2], 2)
                X_post = tf.expand_dims(
                    X[:, :, -1] + X[:, :, -2] - X[:, :, -3], 2
                )
                X_extended = tf.concat(axis=2, values=[X_pre, X, X_post])

                dXi_shape = [self.dims["batch"], 3, self.dims["length"]+1]
                dXi = tf.slice(X_extended, [0, 0, 0], dXi_shape) \
                    - tf.slice(X_extended, [0, 0, 1], dXi_shape)
                Di = tf.sqrt(tf.reduce_sum(tf.square(dXi) + _dynamics_eps, 1))

                # Backbone unit vectors
                Ui = dXi / tf.expand_dims(Di, 1)
                #
                Ui_trans = tf.transpose(Ui, [0, 2, 1])
                z_shape = [self.dims["batch"], self.dims["length"], 3]
                Ui_minus_0 = Ui_trans[:, 0:-1, :]
                Ui_plus_1 = Ui_trans[:, 1:, :]

                # Normals of local reference frames
                n_a = -Ui_minus_0
                n_b = _normed_cross(n_a, Ui_plus_1)
                n_c = tf.cross(n_a, n_b)

                # [B,L,D3] => [B,L,D,3] => [B,L,D]
                dX = tf.reshape(
                    dX, [self.dims["batch"], self.dims["length"], num_vectors, 3]
                )
                dX1, dX2, dX3 = tf.unstack(dX, axis=3)
                # Rotate offsets into local reference frames
                # [B,L,D,1] * [B,L,1,3] => [B,L,D,3]
                dX1, dX2, dX3 = [tf.expand_dims(x, 3) for x in [dX1, dX2, dX3]]
                n_a, n_b, n_c = [tf.expand_dims(x, 2) for x in [n_a, n_b, n_c]]
                dX_rotated = dX1 * n_a + dX2 * n_c + dX3 * n_b

                # Normalize and rescale
                dX_lengths_sq = tf.reduce_sum(tf.square(dX_rotated),3, keep_dims=True)
                dX_lengths = tf.sqrt(dX_lengths_sq + 1E-3)
                L = tf.nn.softplus(tf.expand_dims(L_unc, 3)) * \
                    tf.exp(tf.get_variable(
                    "Scale", (), initializer=tf.constant_initializer(1.)
                ))
                dX_rotated = L * dX_rotated / dX_lengths

                # Global + local offsets
                dX_base = dX_rotated[:, :, 0, :]
                dX_atoms = dX_rotated[:, :, 1:, :]

                # (B,3,L) => (B,L,3)
                X_trans = tf.transpose(X, [0, 2, 1])
                X_full = tf.expand_dims(X_trans + dX_base, 2) + dX_atoms
                coords_shape = [self.dims["batch"],
                                self.dims["length"],
                                self.dims["coords"]]
                coords = tf.reshape(X_full, coords_shape)

        return coords, SS, reconstruction_loss