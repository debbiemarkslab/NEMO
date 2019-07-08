import tensorflow as tf
import numpy as np
from layers import Layers

class Loss:
    def __init__(self, placeholders={}, tensors={}, dims={}, global_step=0, hyperparams={}):
        self.dims = dims
        self.hyperparams = hyperparams
        self.placeholders = placeholders
        self.global_step = global_step
        self.tensors = tensors
        return

    def build_graph(self):
        """Build the computational graph of the loss function"""
        self.layers = Layers(self.placeholders)
        self.masks = self._build_masks()
        self.tensors["seq_1D"], self.tensors["seq_2D"] = self._features_seqs()
        self.tensors["score"] = self._build_score()
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

            # Structure coarse angles
            dihedrals_slice = [-1, self.dims["length"] - 3]
            masks["structure_coarse_angles"] = \
                tf.slice(masks["structure"], [0, 0], dihedrals_slice) \
                * tf.slice(masks["structure"], [0, 1], dihedrals_slice) \
                * tf.slice(masks["structure"], [0, 2], dihedrals_slice) \
                * tf.slice(masks["structure"], [0, 3], dihedrals_slice)
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
            # Distance weights
            # ij_cutoff = 3. + self.placeholders["beta_anneal"] * 300.
            # masks["ij_weights"] = masks["structure_coarse_dists"] * tf.where(
            #             masks["ij_dist"] <= ij_cutoff,
            #             tf.ones_like(masks["ij_dist"]),
            #             tf.zeros_like(masks["ij_dist"])
            # )
            # Short and long split
            ij_cutoff = self.hyperparams["energy"]["backbone_conv"]["width"]
            masks["ij_short"] = masks["structure_coarse_dists"] * tf.where(
                        masks["ij_dist"] <= ij_cutoff,
                        tf.ones_like(masks["ij_dist"]),
                        tf.zeros_like(masks["ij_dist"])
            )
            masks["ij_long"] = masks["structure_coarse_dists"] * \
                (1. - masks["ij_short"])
        return masks

    def _features_seqs(self):
        """ Featurize sequences """
        with tf.variable_scope("Sequences"):
            seqs = self.placeholders["sequences"]
            seqs_1D = tf.expand_dims(seqs, 1)
            # Build 2D-hot encoding of pairwise combinations
            # seqs_flat = tf.reshape(
            #     seqs, 
            #     [self.dims["batch"], self.dims["length"] * self.dims["alphabet"]]
            # )
            # seqs_ij  = tf.expand_dims(seqs_flat, 1) * tf.expand_dims(seqs_flat, 2)
            # seqs_ij = tf.reshape(
            #     seqs_ij, 
            #     [self.dims["batch"], self.dims["length"], self.dims["alphabet"], 
            #      self.dims["length"], self.dims["alphabet"]]
            # )
            # seqs_ij = tf.transpose(seqs_ij, [0, 1, 3, 2, 4])
            # seqs_2D = tf.reshape(seqs_ij,
            #     [self.dims["batch"], self.dims["length"], self.dims["length"],
            #      self.dims["alphabet"] * self.dims["alphabet"]])
            # Tiled and concatenated version
            seqs_i = tf.tile(
                tf.expand_dims(seqs, 1), [1, self.dims["length"], 1, 1]
            )
            seqs_j = tf.tile(
                tf.expand_dims(seqs, 2), [1, 1, self.dims["length"], 1]
            )
            seqs_2D = tf.concat(axis = 3, values = [seqs_i, seqs_j])
        return seqs_1D, seqs_2D

    def _features_atomic_log_distances(self, X):
        """ Compute distances from atomic coordinates """
        d_eps = self.hyperparams["scoring"]["logdists"]["eps"]
        with tf.variable_scope("Distances"):
            B = self.dims["batch"]
            L = self.dims["length"]
            A = self.dims["atoms"]
            # Move length to inner-most dimension for broadcasting
            # [B,L,A3] => [B,LA,3] => [B,LA,LA]
            X = tf.reshape(X, [B, L * A, 3])
            X = tf.transpose(X, [0, 2, 1])
            norm = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
            D = norm + tf.transpose(norm, [0, 2, 1]) \
                - 2 * tf.matmul(X, X, adjoint_a=True)
            logD = 0.5 * tf.log(D + d_eps)
            # [B,LA,LA] => [B,LA,L,A] => [B,L,A,LA] => [B,L,A,L,A]
            logD = tf.reshape(logD, [B, L*A, L, A])
            logD = tf.transpose(logD, perm=[0, 2, 3, 1])
            logD = tf.reshape(logD, [B, L, A, L, A])
            # [B,L,A,L,A] => [B,L,L,A,A] => [B,L,L,AA]
            logD = tf.transpose(logD, perm=[0, 1, 3, 2, 4])
            logD = tf.reshape(logD, [B, L, L, A*A])
            mask = self.masks["structure_coarse_dists"]
            logD *= tf.expand_dims(mask, 3)
        return logD

    def _features_atomic_internal_coords(self, X):
        """ Compute internal coordinates for the full five-atom model.

            Bonding geometry:
                            |         4.SC         | 
                            |          |           | 
                        ... | - 0.N - 1.Ca - 2.C - | ...
                            |                 |    | 
                            |                3.O   | 

            For backbone atoms (N, Ca, C) the four relevant atoms are:
                (1) Backbone atom i of interest
                (2) Backbone atom i-1
                (3) Backbone atom i-2
                (4) Backbone atom i-3
            The internal coordinates are:
                L - Bond length from 1-2
                T - Bond angle 3-2-1
                P - Torsional angle between planes (4,3,2) and (3,2,1)

            For sidechain atoms (O, SC) the four relevant atoms are:
                (1) Side chain atom branched off backbone i
                (2) Backbone atom i+1
                (3) Backbone atom i
                (4) Backbone atom i-1
            The internal coordinates are:
                L - Bond length from 1-3
                T - Bond angle 4-3-1
                P - Torsional angle between planes (4,3,1) and (4,3,2)

        """

        names = ["N", "Ca", "C", "O", "SC"]
        dependencies = {
            "N":  ("backbone", [("N",  0), ("C", -1), ("Ca", -1), ("N",  -1)]),
            "Ca": ("backbone", [("Ca", 0), ("N",  0), ("C",  -1), ("Ca", -1)]),
            "C":  ("backbone", [("C",  0), ("Ca", 0), ("N",   0), ("C",  -1)]),
            "O":  ("side",     [("O",  0), ("N",  1), ("C",   0), ("Ca",  0)]),
            "SC": ("side",     [("SC", 0), ("C",  0), ("Ca",  0), ("N",   0)])
        }
        constants = self.hyperparams["folding"]["constants"]
        _dynamics_eps = constants["xz_eps"]

        def _safe_magnitude(dX):
            dX_square = tf.reduce_sum(tf.square(dX), axis=2, keep_dims=True)
            D = tf.sqrt(dX_square + _dynamics_eps)
            return D

        def _unit_vec(A, B):
            dX = B - A
            D = _safe_magnitude(dX)
            u = dX / D
            return u, D

        def _unit_cross(A, B):
            cross = tf.cross(A, B)
            D = _safe_magnitude(cross)
            return cross / D

        def _internals_backbone(X_0, X_n1, X_n2, X_n3):
            with tf.variable_scope("Backbone"):
                u_minus_0, length = _unit_vec(X_n1, X_0)
                length = length[:, :, 0]
                u_minus_1, _ = _unit_vec(X_n2, X_n1)
                u_minus_2, _ = _unit_vec(X_n3, X_n2)
            with tf.variable_scope("Angles"):
                theta = tf.acos(-tf.reduce_sum(u_minus_1 * u_minus_0, 2))
            with tf.variable_scope("Dihedrals"):
                norms_minus_2 = _unit_cross(u_minus_2, u_minus_1)
                norms_minus_1 = _unit_cross(u_minus_1, u_minus_0)
                phi = tf.sign(tf.reduce_sum(u_minus_2 * norms_minus_1, 2)) \
                    * tf.acos(tf.reduce_sum(norms_minus_2 * norms_minus_1, 2))
            return length, theta, phi

        def _internals_side(X_side, X_p1, X_0, X_n1):
            with tf.variable_scope("Backbone"):
                u_side, length = _unit_vec(X_0, X_side)
                length = length[:, :, 0]
                u_bw, _ = _unit_vec(X_0, X_n1)
                u_fw, _ = _unit_vec(X_0, X_p1)
            with tf.variable_scope("Angles"):
                theta = tf.acos(tf.reduce_sum(u_bw * u_side, 2))
            with tf.variable_scope("Dihedrals"):
                norms_main = _unit_cross(u_bw, u_fw)
                norms_side = _unit_cross(u_side, u_bw)
                phi = tf.sign(tf.reduce_sum(u_fw * norms_side, 2)) \
                    * tf.acos(tf.reduce_sum(norms_main * norms_side, 2))
            return length, theta, phi

        def _slice_atoms(X_set, a_type, a_loc):
            i_start, i_stop = 1+a_loc, 1+a_loc+self.dims["length"]
            a_ix = names.index(a_type)
            a_start, a_stop = a_ix*3, a_ix*3+3
            X_slice = X_set[:, i_start:i_stop, a_start:a_stop]
            return X_slice

        with tf.variable_scope("CartesianToInternal"):
            X_pad = tf.pad(X, [[0, 0], [1, 1], [0, 0]], "CONSTANT")
            X_internal_set = {}
            for atom_ix, name in enumerate(names):
                with tf.variable_scope(name):
                    mode, atoms = dependencies[name]
                    atom_slices = [_slice_atoms(X_pad, t, l) for t, l in atoms]
                    if mode == "backbone":
                        L, T, P = _internals_backbone(*atom_slices)
                    else:
                        L, T, P = _internals_side(*atom_slices)
                    X_internal_set[name] = (L, T, P)

        return X_internal_set

    def _features_hbond_energies(self, X):
        """ Compute the all vs all hbond energies """
        # [B,L,15] => [B,L,5,3] => [B,5,3,L]

        names = ["N", "Ca", "C", "O", "SC"]
        constants = self.hyperparams["folding"]["constants"]
        _dynamics_eps = constants["xz_eps"]

        def _safe_magnitude(dX):
            """ Magnitudes of [B,3,L] vectors """
            dX_square = tf.reduce_sum(tf.square(dX), axis=1, keep_dims=True)
            D = tf.sqrt(dX_square + _dynamics_eps)
            return D

        def _unit_vec(dX):
            """ Normalize the vector dX """
            D = _safe_magnitude(dX)
            u = dX / D
            return u

        def _inv_distance(X_i, X_j):
            """ Inverse distances for [B,3,L] coord sets """
            with tf.variable_scope("InverseDist"):
                dX = tf.expand_dims(X_i, 2) - tf.expand_dims(X_j, 3)
                d_square = tf.reduce_sum(tf.square(dX), axis=1)
                r = 1. / tf.sqrt(d_square + _dynamics_eps)
            return r

        with tf.variable_scope("HBondEnergy"):
            X_typed = tf.reshape(X,
                                 [self.dims["batch"], self.dims["length"],
                                  self.dims["atoms"], 3]
                                 )
            X_typed = tf.transpose(X_typed, [0, 2, 3, 1])
            # Build atom sets [B,3,L]
            X_a = {
                name: X_typed[:, ix, :, :] for ix, name in enumerate(names)
            }
            # Hydrogen built from N(j), Ca(j), and C(j-1)
            X_a["C_prev"] = tf.pad(
                X_a["C"][:, :, 1:], [[0, 0], [0, 0], [0, 1]], "CONSTANT"
            )
            # Place 1 angstrom away from nitrogen in plane
            X_a["H"] = X_a["N"] + _unit_vec(
                _unit_vec(X_a["N"] - X_a["C_prev"])
                + _unit_vec(X_a["N"] - X_a["Ca"])
            )

            # Hydrogen bond energy is relative to DSSP cutoff of 0.5
            energies = 0.5 + (0.084 * 332) * (
                _inv_distance(X_a["O"], X_a["N"])
                + _inv_distance(X_a["C"], X_a["H"])
                - _inv_distance(X_a["O"], X_a["H"])
                - _inv_distance(X_a["C"], X_a["N"])
            )
        return energies

    def _features_coarse_1D(self, X):
        """ Extract target coarse coordinates """
        with tf.variable_scope("CoarseAngles"):
            angle_eps = self.hyperparams["folding"]["constants"]["angle_eps"]
            resid_shape = [self.dims["batch"], 3, self.dims["length"]-1]
            dXi = tf.slice(X, [0, 0, 0], resid_shape) \
                - tf.slice(X, [0, 0, 1], resid_shape)
            Di = tf.sqrt(tf.reduce_sum(tf.square(dXi), 1) + angle_eps)
            dXi_unit = dXi / tf.expand_dims(Di, 1)
            # Dihedrals
            ui_trans = tf.transpose(dXi_unit, [0, 2, 1])
            dihedral_shape = [self.dims["batch"], self.dims["length"]-3, 3]
            r_ji = -tf.slice(ui_trans, [0, 0, 0], dihedral_shape)
            r_jk = tf.slice(ui_trans, [0, 1, 0], dihedral_shape)
            r_kl = tf.slice(ui_trans, [0, 2, 0], dihedral_shape)
            normal_A = tf.cross(r_jk, r_ji)
            normal_B = tf.cross(-r_jk, r_kl)
            mag_A = tf.sqrt(
                tf.reduce_sum(tf.square(normal_A), 2) + angle_eps
            )
            mag_B = tf.sqrt(
                tf.reduce_sum(tf.square(normal_B), 2) + angle_eps
            )
            unit_A = normal_A / tf.expand_dims(mag_A, 2)
            unit_B = normal_B / tf.expand_dims(mag_B, 2)
            # Angle coordinates
            theta = tf.acos(tf.reduce_sum(-r_jk * r_kl, 2))
            phi = tf.sign(tf.reduce_sum(normal_B * r_ji, 2)) \
                * tf.acos(tf.reduce_sum(unit_A * unit_B, 2))
        return theta, phi

    def _build_score(self):
        """Build the graph for model scoring
        """
        with tf.variable_scope("Score"):
            # Compute the score of folded distances
            X_data = self.tensors["coordinates_target"]
            X_model = self.tensors["coordinates_fine"]
            logp_dists = self._score_logp_distances(X_model, X_data)

            # Compute the score of folded distances
            logp_atomic = \
                self._score_logp_internal_coordinates(X_model, X_data)

            # Compute the score of coarse distances and predictions
            logp_coarse, logp_LTP, loss_energy = \
                self._score_logp_coarse(X_model)

            # Compute the log probability of the hbonds
            logp_hbonds = self._score_logp_hbonds(X_model, X_data)

            # Compute the secondary structure loss
            ss_target = self.placeholders["secondary_structure"]
            ss_out = self.tensors["SS"]
            loss_ss = self._score_secondary_structure(ss_out, ss_target)

            # Latent loss
            if self.hyperparams["mode"]["predict_static"]:
                loss_overlap = 0
            else:
                loss_overlap = self._score_latent()

            loss_trajectory = self._score_trajectory()
            loss_weights = self.tensors["loss_weights_physics"] + self.layers.loss

            # Compute the TM score
            av_tm_score = self._score_tm(X_model, X_data)

            # Combine with other module losses
            lams = self.hyperparams["training"]

            loss = -logp_coarse \
                - logp_LTP \
                - logp_dists  \
                - logp_atomic \
                - logp_hbonds \
                - av_tm_score

            # Summaries
            tf.summary.scalar("TrainLoss", tf.squeeze(loss),
                              collections=["gen_batch", "gen_dense"])
            for code in self.hyperparams["test_keys"]:
                tf.summary.scalar("ValidLoss" + code, tf.squeeze(loss),
                                  collections=[code])

            # Additional constraints
            loss += loss_ss \
                + loss_trajectory

            # Physics-specific terms
            if not self.hyperparams["mode"]["predict_static"]:
                loss += self.tensors["loss_init_dX"] \
                    + self.tensors["loss_init_RG"] \
                    + loss_energy \
                    + loss_overlap

                tf.summary.scalar("LossInitRG", self.tensors["loss_init_RG"],
                              collections=["gen_batch", "gen_dense"])
                tf.summary.scalar("LossInitdX", self.tensors["loss_init_dX"],
                              collections=["gen_batch", "gen_dense"])
                tf.summary.scalar("LossDynamics", self.tensors["loss_dynamics"],
                              collections=["gen_batch", "gen_dense"])
                tf.summary.scalar("LossInterfaces", self.tensors["loss_interfaces"],
                              collections=["gen_batch", "gen_dense"])

            tf.summary.scalar("TrainLossFull", loss,
                              collections=["gen_batch", "gen_dense"])
            tf.summary.scalar("LossWeights", loss_weights,
                              collections=["gen_batch", "gen_dense"])
        return loss

    def _score_latent(self):
        """ Regularization of latent information """

        def covariance_loss(X, N_samples):
            """ Compute variance of eigenvalues of covariance matrix 

                No longer necessary post batch-renormalization
            """
            N_channels = tf.to_float(X.get_shape().as_list()[1])
            X_mean = tf.reduce_sum(X, 1, keep_dims=True) / N_channels
            # Covariance loss
            # X_mean = tf.reduce_sum(X, 0, keep_dims=True) / N_samples
            # X = X - X_mean
            # S = tf.matmul(X,X,transpose_a=True)
            # Cov = S / N_samples
            # Cov_shrink = (tf.diag(tf.diag_part(Cov)) + S) / (N_samples + 1)
            # e,_ = tf.self_adjoint_eig(Cov_shrink)
            # e_mean, e_var = tf.nn.moments(e, axes=[0])
            # loss = tf.reduce_mean(tf.square(e - 1.) + tf.square(X_mean))
            loss = tf.reduce_mean(tf.square(X_mean))
            return loss

        # Node channel decorrelation
        num_positions = tf.reduce_sum(self.masks["seqs"])
        num_channels = self.tensors["nodes"].get_shape().as_list()[3]
        data_nodes = tf.reshape(
            self.tensors["nodes"], 
            [self.dims["batch"] * self.dims["length"], num_channels]
        )
        loss_nodes = covariance_loss(data_nodes, num_positions)

        # Edge channel decorrelation
        min_length = tf.reduce_min(self.placeholders["lengths"])
        num_positions = tf.to_float(self.dims["batch"] * min_length)
        rand_ix = tf.random_uniform((), minval=0, maxval=min_length-1, dtype=tf.int32)
        edge_slice = self.tensors["edges"][:,rand_ix,0:min_length,:]
        num_channels = edge_slice.get_shape().as_list()[2]
        data_edges = tf.reshape(
            edge_slice, 
            [self.dims["batch"] * min_length, num_channels]
        )
        loss_edges = covariance_loss(data_edges, num_positions)

        tf.summary.scalar("LossNodes", loss_nodes,
                              collections=["gen_batch", "gen_dense"])
        tf.summary.scalar("LossEdges", loss_edges,
                              collections=["gen_batch", "gen_dense"])
        loss_overlap = loss_nodes + loss_edges
        loss_overlap *= self.hyperparams["scoring"]["overlap"]["weight"]
        return loss_overlap

    def _score_flatten_coords(self, X):
        """Extract the coordinate sets"""
        with tf.variable_scope("FlattenCoords"):
            # Move length to inner-most dimension for broadcasting
            length = self.dims["atoms"] * self.dims["length"]
            coord_set = tf.reshape(X, [self.dims["batch"], length, 3])
            coord_set = tf.transpose(coord_set, [0, 2, 1])
        return coord_set

    def _score_secondary_structure(self, ss, ss_target):
        with tf.variable_scope("PredictedSS_lrA"):
            # Pre-folding SS prediction
            ss_pre_logits = self.tensors["ss_pre_logits"]
            ss_pre = tf.nn.softmax(ss_pre_logits)
            ss_post = tf.nn.softmax(ss)
            ss_pre_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ss_target,logits=ss_pre_logits)
            ss_pre_loss = tf.reduce_sum(self.masks["ss"]* ss_pre_loss) / tf.reduce_sum(self.masks["ss"])
            ss_post_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ss_target,logits=ss)
            ss_post_loss = tf.reduce_sum(self.masks["ss"]* ss_post_loss) / tf.reduce_sum(self.masks["ss"])

            def _ss_summary(name, data, collection="gen_dense"):
                """ Secondary structure image """
                tf.summary.image(
                    name,
                    tf.expand_dims(tf.transpose(data, perm=[0, 2, 1]), 3),
                    collections=[collection]
                )
                return

            ss_image = tf.concat(
                axis=2,
                values = [ss_pre, ss_post, ss_target]
            )
            # 3-state predictions
            ss_log_probs = 0.
            if self.hyperparams["mode"]["predict_ss3"]:
                ss_log_probs = tf.reduce_sum(self.tensors["SS_3_log_probs"]) \
                    / tf.reduce_sum(self.masks["seqs"])
                ss3_marginals = tf.nn.softmax(self.tensors["SS_3_log_marginals"])
                tf.summary.scalar("SS3_logprob", ss_log_probs, collections=["gen_dense"])
                ss_image = tf.concat(
                    axis=2,
                    values = [ss3_marginals, self.tensors["SS_3_sample"], 
                              self.tensors["SS_3_target"], ss_image]
                )
                for code in self.hyperparams["test_keys"]:
                    tf.summary.scalar("SS3_logprob" + code, ss_log_probs, collections=[code])

            # Image summaries
            _ss_summary("SS_image", ss_image)
            for code in self.hyperparams["test_keys"]:
                _ss_summary("SS_image" + code, ss_image, code)

            # Images for checking
            ss_loss = ss_pre_loss + ss_post_loss - ss_log_probs

            for code in self.hyperparams["test_keys"]:
                tf.summary.scalar("SSPre" + code, ss_pre_loss, collections=[code])
                tf.summary.scalar("SSPost" + code, ss_post_loss, collections=[code])
            tf.summary.scalar("SSPre", ss_pre_loss, collections=["gen_dense"])
            tf.summary.scalar("SSPost", ss_post_loss, collections=["gen_dense"])
        return ss_loss

    def _score_trajectory(self):
        """ Autocorrelation loss """
        with tf.variable_scope("TrajectoryLoss"):
            # Trajectory is [B, T, 3, L]
            if self.hyperparams["mode"]["predict_static"]:
                time_indices = [0]
            else:
                time_indices = tf.range(
                    self.placeholders["langevin_steps"], delta=5
                )
            d_eps = self.hyperparams["scoring"]["logdists"]["eps"]
            L2_eps = 1E-2
            
            with tf.variable_scope("TargetFeatures"):
                X_target = self.tensors["coarse_target"]
                # Build distances
                # (N, 3, L, 1) - (N, 3, 1, L)  => (N, 3, L, L)
                dX_target = tf.expand_dims(X_target, 3) - tf.expand_dims(X_target, 2)
                D = tf.sqrt(tf.reduce_sum(tf.square(dX_target), 1) + d_eps)
                # Build local reference frame
                X_trans = tf.transpose(X_target, [0,2,1])
                u_i = tf.nn.l2_normalize(X_trans[:,1:,:] - X_trans[:,:-1,:], 2, epsilon=L2_eps)
                u_bw = u_i[:,:-1,:]
                u_fw = u_i[:,1:,:]
                e_i = tf.nn.l2_normalize(u_bw - u_fw, 2, epsilon=L2_eps)
                a_i = tf.nn.l2_normalize(tf.cross(u_bw, u_fw), 2, epsilon=L2_eps)
                exa_i = tf.cross(e_i, a_i)
                R = tf.stack([e_i, a_i, exa_i], 2)
                R = tf.pad(R, [[0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")
                # (N, 3, L, L) => (N, L, L, 3)
                dX_trans = tf.transpose(dX_target, [0,2,3,1])
                # (N, L, 1, 3, 3) * (N, L, L, 1, 3) => (N, L, L, 3)
                # This contraction was faster than whatever code tf.einsum created
                orient_target = tf.reduce_sum(tf.expand_dims(R, 2) * tf.expand_dims(dX_trans, 3), 4)

            with tf.variable_scope("ModelFeatures"):
                X_samples = tf.gather(self.tensors["trajectory"], time_indices, axis=1)
                # (B,T,L,3)
                dX = tf.expand_dims(X_samples, 4) - tf.expand_dims(X_samples, 3)
                dXsq = tf.square(dX)
                logD = 0.5 * tf.log(tf.reduce_sum(dXsq, 2) + d_eps)
                # Build donor & acceptor vectors in local reference frame
                L2_eps = 1E-2
                # (B,T,3,L) => (B,T,L,3)
                Xt_trans = tf.transpose(X_samples, [0,1,3,2])
                u_i = tf.nn.l2_normalize(Xt_trans[:,:,1:,:] - Xt_trans[:,:,:-1,:], 3, epsilon=L2_eps)
                u_bw = u_i[:,:,:-1,:]
                u_fw = u_i[:,:,1:,:]
                e_i = tf.nn.l2_normalize(u_bw - u_fw, 3, epsilon=L2_eps)
                a_i = tf.nn.l2_normalize(tf.cross(u_bw, u_fw), 3, epsilon=L2_eps)
                exa_i = tf.cross(e_i, a_i)
                # 3x(B,T,L,3) => (B,T,L,3,3)
                R = tf.stack([e_i, a_i, exa_i], 3)
                R = tf.pad(R, [[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]], "CONSTANT")
                # Unit difference vectors
                # (B,T,3,L,L) => (B,T,L,L,3)
                dX_trans = tf.transpose(dX, [0,1,3,4,2])
                # (N, T, L, 1, 3, 3) * (N, T, L, L, 1, 3) => (N, T, L, L, 3)
                orient_model = tf.reduce_sum(tf.expand_dims(R, 3) * tf.expand_dims(dX_trans, 4), 5)

            # Orientation image summaries
            image_mask = tf.expand_dims(self.masks["dists"], 3)
            tf.summary.image("OrientModel", image_mask * orient_model[:,-1,:,:,:], collections=["gen_dense"])
            tf.summary.image("OrientData", image_mask * orient_target, collections=["gen_dense"])

            # Build time mask
            Dt_mask = tf.expand_dims(self.masks["dists"], 1) * tf.ones_like(logD)

            # Target geometry scoring
            mask_1D = self.masks["structure"]
            mask_1D = tf.pad(
                mask_1D[:,:-2] * mask_1D[:,1:-1] * mask_1D[:,2:], [[0, 0], [1, 1]], "CONSTANT"
            )
            mask_2D = tf.expand_dims(tf.expand_dims(mask_1D, 1) * tf.expand_dims(mask_1D, 2), 1)
            mask_2D *= tf.expand_dims(self.masks["structure_coarse_dists"], 1)
            logD_target = tf.expand_dims(self.tensors["coarse_target_logD"], 1)
            Dt_mask_structure = Dt_mask * mask_2D
            Dt_mask_dists = Dt_mask * tf.expand_dims(self.masks["structure_coarse_dists"], 1)
            # orient_error = tf.abs(2. - tf.reduce_sum(tf.expand_dims(orient_target, 1) * orient_model, 4))
            D_target = tf.exp(logD_target)
            D_model = tf.exp(logD)
            # Measure their distance by local alignment
            offset_error = tf.reduce_sum(tf.square(orient_model - tf.expand_dims(orient_target, 1)), 4)
            target_error = 0.5 * (Dt_mask_structure * offset_error + Dt_mask_dists * tf.square(D_model - D_target))

            # Pooled reduction
            error_shape = tf.shape(target_error)
            B = error_shape[0]
            T = error_shape[1]
            L = self.dims["length"]
            error_ij = tf.reshape(target_error, [B*T, L, L])
            mask_flat = tf.reshape(Dt_mask_dists, [B*T, L, L])
            pool_width = self.hyperparams["energy"]["backbone_conv"]["width"]
            error_ij_pooled = self.layers.pool_2D(error_ij, mask_flat, pool_width)
            error_ii_pooled = tf.matrix_diag_part(error_ij_pooled)
            error_ij_pooled = 0.5 * error_ij_pooled + 0.25 * (
                tf.expand_dims(error_ii_pooled, 1)
                + tf.expand_dims(error_ii_pooled, 2)
            )
            target_error = tf.sqrt(error_ij_pooled + 1E-5)
            target_error = tf.reshape(target_error, [B, T, L, L])

            # Contact focused averages
            logD_min = tf.minimum(logD, logD_target)
            # logD_min = logD
            contact_cutoff_D = \
                self.hyperparams["scoring"]["logdists"]["contact"]
            contact_precision = \
                self.hyperparams["scoring"]["logdists"]["contact_precision"]
            mask_contacts = tf.nn.sigmoid(
                contact_precision * (np.log(contact_cutoff_D) - logD_min)
            ) * Dt_mask_dists

            error_distance = tf.reduce_sum(mask_contacts * target_error) / tf.reduce_sum(mask_contacts)
            tf.summary.scalar(
                "TrajectoryDistanceLoss", error_distance,
                collections=["gen_batch", "gen_dense"]
            )

            if self.hyperparams["mode"]["predict_static"]:
                loss_sensitivity = 0
            else:
                # Compute variances of trajectory subsamples
                T = tf.to_float(tf.shape(X_samples)[1])
                T_1 = tf.to_int32(T / 3.)
                T_2 = tf.to_int32(T * 2. / 3.)
                T_3 = tf.to_int32(T)

                def _var_logD(T_A, T_B):
                    X = logD[:,T_A:T_B,:,:]
                    mask = Dt_mask[:,T_A:T_B,:,:]
                    mean = tf.reduce_sum(mask * X, 1, keep_dims=True) \
                        / (tf.reduce_sum(mask, 1, keep_dims=True) + 1E-4)
                    dXsq = tf.square(X - mean)
                    var = tf.reduce_sum(mask * dXsq, 1) \
                        / (tf.reduce_sum(mask, 1) + 1E-4)
                    return var

                var_2 = _var_logD(T_1, T_2)
                var_3 = _var_logD(T_2, T_3)
                var_23 = _var_logD(T_1, T_3)
                var_avg = 0.5 * (var_2 + var_3)

                # Potential reduction scale factor
                mask = self.masks["dists"]
                Rhat = var_23 / (var_avg + (1. - mask))
                Rhat = tf.reduce_sum(mask * Rhat, [1,2]) \
                    / tf.reduce_sum(mask, [1,2])
                Var = tf.reduce_sum(mask * var_avg, [1,2]) \
                    / tf.reduce_sum(mask, [1,2])

                Rhat = tf.reduce_mean(Rhat)
                Var = tf.reduce_mean(Var)
                tf.summary.scalar(
                    "Rhat", Rhat,
                    collections=["gen_batch", "gen_dense"]
                )
                tf.summary.scalar(
                    "Var", Var,
                    collections=["gen_batch", "gen_dense"]
                )

                # Autocorrelation stuff
                # Compute time-lagged statistics
                T_range = tf.minimum(T_3-T_1-1,5)
                lag = tf.random_uniform((), minval=1, maxval=T_range, dtype=tf.int32)
                # logD = logD[:,T_1:T_3,:,:]
                # Dt_mask = Dt_mask[:,T_1:T_3,:,:]
                lag1_mask = Dt_mask[:,lag:,:,:] * Dt_mask[:,:-lag,:,:]
                lag1_dist = tf.square(logD[:,lag:,:,:] - logD[:,:-lag,:,:])
                lag1_dist = tf.reduce_sum(lag1_mask * lag1_dist, [1,2,3]) \
                    / tf.reduce_sum(lag1_mask, [1,2,3])
                loss_lag1 = lag1_dist
                loss_lag1 = 3 * -tf.reduce_mean(loss_lag1)
                
                tf.summary.scalar(
                    "LossLag", loss_lag1,
                    collections=["gen_batch", "gen_dense"]
                )

                # Sensitivity loss
                X_t = self.tensors["trajectory"]
                eps = tf.transpose(
                    self.tensors["retro_perturbations"],
                    [1,0,2,3]
                )
                mask_expand = tf.expand_dims(self.masks["seqs"], 1)
                # Compute norms of first and second differences
                dX_t = X_t[:,1:,:,:] - X_t[:,:-1,:,:]
                deps = eps[:,1:,:,:] - eps[:,:-1,:,:]
                dX_t_sq = tf.reduce_sum(tf.square(dX_t), 2)
                deps_sq = tf.reduce_sum(tf.square(deps), 2)
                # Distances are is [B, T-1]
                logD_dX_t = 0.5 * tf.log(
                    tf.reduce_sum(mask_expand * dX_t_sq, 2) + 1E-8
                )
                logD_deps_t = 0.5 * tf.log(
                    tf.reduce_sum(mask_expand * deps_sq, 2) + 1E-8
                )
                loss_sensitivity = 10. * tf.nn.relu(
                    logD_dX_t - logD_deps_t
                )
                retro_mask = tf.transpose(self.tensors["retro_steps"], [1,0])
                retro_mask = retro_mask[:, :-1]
                loss_sensitivity = tf.reduce_sum(retro_mask * loss_sensitivity) \
                    / tf.reduce_sum(retro_mask)
                tf.summary.scalar(
                    "LossSensitivity", loss_sensitivity,
                    collections=["gen_batch", "gen_dense"]
                )

            loss_trajectory = loss_sensitivity + error_distance
        return loss_trajectory


    def _score_energy(self, X_model, X_data):
        """ Energy loss """
        def _energy(X, stop_gradient=False):
            """ Compute the structure-masked energy """
            fields = self.tensors["energy"]
            # key = "solvent_readonly" if stop_gradient else "solvent"
            # for (coeff, scale, scaled_coeff, radii) in fields[key]:
            #     Uij = Uij + coeff * tf.nn.sigmoid(scale * (radii - D))
            with tf.variable_scope("BackboneFeatures"):
                # Compute internal coordinates
                L, T, P = self._score_coarse_internal(X)
                # Pad and expand
                L = tf.pad(L, [[0, 0], [3, 0]], "CONSTANT")
                T = tf.pad(T, [[0, 0], [3, 0]], "CONSTANT")
                P = tf.pad(P, [[0, 0], [3, 0]], "CONSTANT")
                angle_mask = self.masks["structure_coarse_angles"]
                angle_mask = tf.pad(angle_mask, [[0, 0], [3, 0]], "CONSTANT")
                angle_mask = tf.expand_dims(tf.expand_dims(angle_mask, 1), 3)
                # Restricted Boltzmann Machine governs backbone angles
                # key = "backbone_conv_readonly" if stop_gradient else "backbone_conv"
                # W, b_h, b_v, L_loc, L_prec, h_mask = fields[key]
                key = "backbone_conv_readonly" if stop_gradient else "backbone_conv"
                # W, b_h, b_v, L_loc, L_prec, h_mask = fields[key]
                W1, W2, W3, b1, b2, b3, c3, b_v, L_loc, L_prec, h_mask = fields[key]
                # h_mask *= angle_mask
                # Unit vectors encode angles
                v_TP = tf.stack([tf.cos(T), tf.sin(T) * tf.cos(P), tf.sin(T) * tf.sin(P)], 2)
                # Radial basis functions encode lengths
                v_L = tf.exp(-L_prec * tf.square(tf.expand_dims(L,2) - L_loc))
                v_1D = tf.expand_dims(tf.concat(axis=2, values = [v_TP, v_L]), 1) * angle_mask
            # Pairwise calculation
            dX = tf.expand_dims(X, 3) - tf.expand_dims(X, 2)
            D = tf.sqrt(tf.reduce_sum(tf.square(dX), 1) + 0.1)
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
                mask = self.masks["ij_long"]
                mask_expand = tf.expand_dims(mask, 3)
                v_D = tf.exp(-D_prec * tf.square(tf.expand_dims(D,3) - D_loc))
                v_D = mask_expand * tf.concat(axis=3, values=[v_D, v_R])
                h1_2D = tf.nn.softplus(tf.nn.convolution(
                    v_D, filter=W1, strides=(2, 2),
                    padding="SAME", dilation_rate=(1, 1)
                ) + b1)
                h2_2D = tf.nn.softplus(tf.nn.convolution(
                    h1_2D, filter=W2, strides=(2, 2),
                    padding="SAME", dilation_rate=(1, 1)
                ) + b2) * tf.nn.avg_pool(
                        value=mask_expand,
                        ksize=[1, 1, 1, 1], 
                        strides=[1, 4, 4, 1], 
                        padding="SAME"
                )
                Uij_conv = -tf.reduce_sum(c2 * h2_2D, [1,2,3]) - tf.reduce_sum(c0 * v_D, [1,2,3])
                Uij = Uij_conv
            with tf.variable_scope("BackboneConv"):
                key = "backbone_conv_readonly" if stop_gradient else "backbone_conv"
                # W, b_h, b_v, L_loc, L_prec, h_mask = fields[key]
                W1, W2, W3, b1, b2, b3, c3, b_v, L_loc, L_prec, h_mask = fields[key]
                # Unit vectors encode angles
                h1_1D = tf.nn.softplus(tf.nn.convolution(
                        v_1D, filter=W1, strides=(2, 2),
                        padding="SAME", dilation_rate=(1, 1)
                    ) + b1)
                h2_1D = tf.nn.softplus(tf.nn.convolution(
                    h1_1D, filter=W2, strides=(2, 2),
                    padding="SAME", dilation_rate=(1, 1)
                ) + b2)
                # Aggregate 2D features for 1D
                v_h2_2D = tf.reduce_sum(h2_2D, 1, keep_dims=True)
                v_h2_2D = tf.nn.l2_normalize(v_h2_2D, 3, epsilon=1E-3)
                h2_1D = tf.concat(axis=3,values=[h2_1D, v_h2_2D])
                h3_1D = c3 * tf.nn.softplus(tf.nn.convolution(
                    h2_1D, filter=W3, strides=(2, 2),
                    padding="SAME", dilation_rate=(1, 1)
                ) + b3)
                Ui = -tf.reduce_sum(h3_1D, [1,2,3]) \
                     -tf.reduce_sum(b_v * v_1D, [1,2,3])
            beta = self.tensors["beta"]
            if stop_gradient:
                beta = tf.stop_gradient(beta)
            U = beta * (Ui + Uij)
            return U

        # Energy difference loss
        std = 0.01
        X_model = tf.cond(
            self.placeholders["training"],
            lambda: X_model + std * tf.random_normal(tf.shape(X_model)),
            lambda: X_model
        )
        X_data = tf.cond(
            self.placeholders["training"],
            lambda: X_data + std * tf.random_normal(tf.shape(X_data)),
            lambda: X_data
        )
        U_model = _energy(tf.stop_gradient(X_model))
        U_data = _energy(tf.stop_gradient(X_data))
        loss_CD = U_data - U_model
        U_model_stop = _energy(X_model, stop_gradient=True)

        N = tf.reduce_sum(self.masks["seqs"])
        loss_CD = tf.reduce_sum(loss_CD) / N
        avgU_final = tf.reduce_sum(U_model_stop) / N

        # Expose energies to save
        self.tensors["energy_model"] = U_model
        self.tensors["energy_data"] = U_data

        avgU_init = self.tensors["loss_init"] / N
        energy_loss = tf.cond(
            loss_CD > 0,
            lambda: loss_CD,
            lambda: 0.
        )
        tf.summary.scalar(
            "LossEnergy", energy_loss, collections=["gen_batch", "gen_dense"]
        )
        tf.summary.scalar(
            "LossCD", loss_CD, collections=["gen_batch", "gen_dense"]
        )
        tf.summary.scalar(
            "EnergyInit", avgU_init, collections=["gen_batch", "gen_dense"]
        )
        tf.summary.scalar(
            "EnergyFinal", avgU_final, collections=["gen_batch", "gen_dense"]
        )
        return energy_loss

    def _score_coarse_internal(self, X):
        """ Extract internal coordinates """
        angle_eps = self.hyperparams["folding"]["constants"]["angle_eps"]
        batch_size = tf.shape(X)[0]
        resid_shape = [batch_size, 3, self.dims["length"]-1]
        def _normed_cross(A, B):
            cross = tf.cross(A, B)
            mag = tf.sqrt(
                tf.reduce_sum(tf.square(cross), 2,
                              keep_dims=True) + angle_eps
            )
            return cross / mag
        dXi = tf.slice(X, [0, 0, 1], resid_shape) \
            - tf.slice(X, [0, 0, 0], resid_shape)
        Di = tf.sqrt(tf.reduce_sum(tf.square(dXi), 1) + angle_eps)
        dXi_unit = dXi / tf.expand_dims(Di, 1)
        # Dihedrals
        ui_trans = tf.transpose(dXi_unit, [0, 2, 1])
        dihedral_shape = [batch_size, self.dims["length"]-3, 3]
        u_minus_2 = tf.slice(ui_trans, [0, 0, 0], dihedral_shape)
        u_minus_1 = tf.slice(ui_trans, [0, 1, 0], dihedral_shape)
        u_minus_0 = tf.slice(ui_trans, [0, 2, 0], dihedral_shape)
        norms_minus_2 = _normed_cross(u_minus_2, u_minus_1)
        norms_minus_1 = _normed_cross(u_minus_1, u_minus_0)
        # Angle coordinates
        theta =  tf.acos(-tf.reduce_sum(u_minus_1 * u_minus_0, 2))
        phi = tf.sign(tf.reduce_sum(u_minus_2 * norms_minus_1, 2)) \
            * tf.acos(tf.reduce_sum(norms_minus_2 * norms_minus_1, 2))
        Di = Di[:,2:]
        return Di, theta, phi

    def _score_logp_coarse(self, X_model):
        """ Extract target coarse coordinates """
        def _summary(X):
            return tf.expand_dims(tf.expand_dims(X, 0), 3)

        def _get_scale_param(name, init):
            with tf.variable_scope("Scale_" + name):
                scale = tf.get_variable(
                    name, (), initializer=tf.constant_initializer(init)
                )
                scale_constrained = tf.nn.softplus(scale)
            return scale_constrained

        def _logp_vonmises_fisher(T_model, T_data, P_model, P_data, kappa,
                                  mask):
            """ Von Mises-Fisher distribution """
            with tf.variable_scope("AnglesVMF"):
                log_2sinh_kappa = kappa + tf.log(1.0 - tf.exp(-kappa))
                logp_TP = kappa * tf.cos(T_model) * tf.cos(T_data) \
                    + tf.sin(T_model) * tf.sin(T_data) \
                    * tf.cos(P_model - P_data) \
                    + tf.log(kappa) - log_2sinh_kappa - np.log(2.0 * np.pi)
                # Average per position
                logp_TP = tf.reduce_sum(mask * logp_TP) / tf.reduce_sum(mask)
            return logp_TP

        with tf.variable_scope("LogPCoarse"):
            # Pick last atom type (side chain center of mass)
            X_data = self.tensors["coarse_target"]
            X_model = self.tensors["coordinates_coarse"]
            with tf.variable_scope("Distances"):
                # Distance Losses
                d_eps = self.hyperparams["scoring"]["logdists"]["eps"]
                logD_data = \
                    0.5 * tf.log(self._score_squared_dists(X_data) + d_eps)
                logD_model = \
                    0.5 * tf.log(self._score_squared_dists(X_model) + d_eps)
            with tf.variable_scope("Packing"):
                # Compute packing statistics
                def _density(logD):
                    # Soft contact density function
                    C = tf.nn.sigmoid(np.log(8.0) - logD)
                    log_density = tf.log(tf.reduce_sum(
                        self.masks["structure_coarse_dists"] * C, 2
                    ) + d_eps)
                    return log_density
                density_data = _density(logD_data)
                density_model = _density(logD_model)
                l1_density = -tf.abs(density_data - density_model)
                mask_linear = self.masks["structure"]
                density_loss = tf.reduce_sum(mask_linear * l1_density) \
                    / tf.reduce_sum(mask_linear)
                logp_density = density_loss
            with tf.variable_scope("Angles"):
                L_model, T_model, P_model = self._score_coarse_internal(X_model)
                L_data, T_data, P_data = self._score_coarse_internal(X_data)
            with tf.variable_scope("CoarseLogP"):
                mask = self.masks["structure_coarse_dists"]
                # residual = logD_data - logD_model
                residual = tf.exp(logD_data) - tf.exp(logD_model)
                error_ij = tf.square(residual)
                contact_cutoff_D = self.hyperparams["scoring"]["logdists"]["contact"]
                contact_precision = self.hyperparams["scoring"]["logdists"]["contact_precision"]
                logD_min = tf.minimum(logD_model, logD_data)
                mask_full = self.masks["structure_coarse_dists"]
                mask_contacts = tf.nn.sigmoid(
                    contact_precision * (np.log(contact_cutoff_D) - logD_min)
                ) * mask_full
                # Pool error over windowed subsystem around ij
                pool_width = self.hyperparams["energy"]["backbone_conv"]["width"]
                error_ij_pooled = self.layers.pool_2D(error_ij, mask_full, pool_width)
                error_ii_pooled = tf.matrix_diag_part(error_ij_pooled)
                error_ij_pooled = 0.5 * error_ij_pooled + 0.25 * (
                    tf.expand_dims(error_ii_pooled, 1)
                    + tf.expand_dims(error_ii_pooled, 2)
                )
                error_ij_pooled = tf.sqrt(error_ij_pooled + 1E-5)
                logp_protein = -tf.reduce_sum(mask_contacts * error_ij_pooled, [1,2]) \
                    / tf.reduce_sum(mask_contacts, [1,2])
                logp_D = tf.reduce_mean(logp_protein)

                # L1 Internal coordinate loss
                dX_model = tf.stack(
                    [tf.cos(T_model), 
                     tf.sin(T_model) * tf.cos(P_model), 
                     tf.sin(T_model) * tf.sin(P_model)], 1
                )
                dX_data = tf.stack(
                    [tf.cos(T_data), 
                     tf.sin(T_data) * tf.cos(P_data), 
                     tf.sin(T_data) * tf.sin(P_data)], 1
                )
                dX_dists = tf.reduce_sum(tf.square(dX_data - dX_model), 1)
                angles_mask = self.masks["structure_coarse_angles"]
                error_L = tf.square(tf.log(L_data + 1E-3) - tf.log(L_model + 1E-3))
                pool_width = self.hyperparams["energy"]["backbone_conv"]["width"]
                dX_pooled = self.layers.pool_1D(dX_dists + error_L, angles_mask, pool_width)
                dX_pooled = tf.sqrt(dX_pooled + 1E-5)
                logp_LTP = -tf.reduce_sum(angles_mask * dX_pooled) \
                    / tf.reduce_sum(angles_mask)
                # Initially focus only on angles and lengths
                logp = logp_LTP + logp_D
            if self.hyperparams["mode"]["predict_static"]:
                loss_energy = 0
            else:
                with tf.variable_scope("Energy"):
                    loss_energy = self._score_energy(X_model, X_data)
            # Summaries
            with tf.variable_scope("Summaries"):
                tf.summary.scalar("LogPDists", logp_D,
                                  collections=["gen_batch", "gen_dense"])
                tf.summary.scalar("LogPDensity", logp_density,
                                  collections=["gen_batch", "gen_dense"])
                tf.summary.scalar("LogPAngles", logp_LTP,
                                  collections=["gen_batch", "gen_dense"])
                # Contact densities
                image_density_data = tf.expand_dims(
                    tf.expand_dims(mask_linear * density_data, 0),3
                )
                image_density_model = tf.expand_dims(
                    tf.expand_dims(mask_linear * density_model, 0),3
                )
                # Images
                tf.summary.image("DensityModel", image_density_model,
                                 collections=["gen_dense"])
                tf.summary.image("DensityTarget", image_density_data,
                                 collections=["gen_dense"])
                # Train summaries
                tf.summary.image("LogDData",
                                 tf.expand_dims(mask * logD_data, 3),
                                 collections=["gen_dense"])
                tf.summary.image("LogDModel",
                                 tf.expand_dims(mask * logD_model, 3),
                                 collections=["gen_dense"])
                # Angle summaries
                def _angles_to_color(T, P):
                    return tf.expand_dims(tf.stack(
                        [tf.cos(T), tf.sin(T) * tf.cos(P), 
                         tf.sin(T) * tf.sin(P)], 2
                    ) * tf.expand_dims(angles_mask, 2), 0)
                tf.summary.image(
                    "AnglesTarget", _angles_to_color(T_data, P_data),
                    collections=["gen_dense"]
                )
                tf.summary.image(
                    "AnglesModel", _angles_to_color(T_model, P_model),
                    collections=["gen_dense"]
                )
        return logp_D, logp_LTP, loss_energy

    def _score_full_cartesian_to_internal(self, X_cart):
        """ Compute internal coordinates for the full five-atom model.

            Bonding geometry:
                            |         4.SC         | 
                            |          |           | 
                        ... | - 0.N - 1.Ca - 2.C - | ...
                            |                 |    | 
                            |                3.O   | 

            For backbone atoms (N, Ca, C) the four relevant atoms are:
                (1) Backbone atom i of interest
                (2) Backbone atom i-1
                (3) Backbone atom i-2
                (4) Backbone atom i-3
            The internal coordinates are:
                L - Bond length from 1-2
                T - Bond angle 3-2-1
                P - Torsional angle between planes (4,3,2) and (3,2,1)

            For sidechain atoms (O, SC) the four relevant atoms are:
                (1) Side chain atom branched off backbone i
                (2) Backbone atom i+1
                (3) Backbone atom i
                (4) Backbone atom i-1
            The internal coordinates are:
                L - Bond length from 1-3
                T - Bond angle 4-3-1
                P - Torsional angle between planes (4,3,1) and (4,3,2)

        """

        names = ["N", "Ca", "C", "O", "SC"]
        dependencies = {
            "N":  ("backbone", [("N",  0), ("C", -1), ("Ca", -1), ("N",  -1)]),
            "Ca": ("backbone", [("Ca", 0), ("N",  0), ("C",  -1), ("Ca", -1)]),
            "C":  ("backbone", [("C",  0), ("Ca", 0), ("N",   0), ("C",  -1)]),
            "O":  ("side",     [("O",  0), ("N",  1), ("C",   0), ("Ca",  0)]),
            "SC": ("side",     [("SC", 0), ("C",  0), ("Ca",  0), ("N",   0)])
        }
        constants = self.hyperparams["folding"]["constants"]
        _dynamics_eps = constants["xz_eps"]

        def _safe_magnitude(dX):
            dX_square = tf.reduce_sum(tf.square(dX), axis=2, keep_dims=True)
            D = tf.sqrt(dX_square + _dynamics_eps)
            return D

        def _unit_vec(A, B):
            dX = B - A
            D = _safe_magnitude(dX)
            u = dX / D
            return u, D

        def _unit_cross(A, B):
            cross = tf.cross(A, B)
            D = _safe_magnitude(cross)
            return cross / D

        def _internals_backbone(X_0, X_n1, X_n2, X_n3):
            with tf.variable_scope("Backbone"):
                u_minus_0, length = _unit_vec(X_n1, X_0)
                length = length[:, :, 0]
                u_minus_1, _ = _unit_vec(X_n2, X_n1)
                u_minus_2, _ = _unit_vec(X_n3, X_n2)
            with tf.variable_scope("Angles"):
                theta = tf.acos(-tf.reduce_sum(u_minus_1 * u_minus_0, 2))
            with tf.variable_scope("Dihedrals"):
                norms_minus_2 = _unit_cross(u_minus_2, u_minus_1)
                norms_minus_1 = _unit_cross(u_minus_1, u_minus_0)
                phi = tf.sign(tf.reduce_sum(u_minus_2 * norms_minus_1, 2)) \
                    * tf.acos(tf.reduce_sum(norms_minus_2 * norms_minus_1, 2))
            return length, theta, phi

        def _internals_side(X_side, X_p1, X_0, X_n1):
            with tf.variable_scope("Backbone"):
                u_side, length = _unit_vec(X_0, X_side)
                length = length[:, :, 0]
                u_bw, _ = _unit_vec(X_0, X_n1)
                u_fw, _ = _unit_vec(X_0, X_p1)
            with tf.variable_scope("Angles"):
                theta = tf.acos(tf.reduce_sum(u_bw * u_side, 2))
            with tf.variable_scope("Dihedrals"):
                norms_main = _unit_cross(u_bw, u_fw)
                norms_side = _unit_cross(u_side, u_bw)
                phi = tf.sign(tf.reduce_sum(u_fw * norms_side, 2)) \
                    * tf.acos(tf.reduce_sum(norms_main * norms_side, 2))
            return length, theta, phi

        def _slice_atoms(X_set, a_type, a_loc):
            i_start, i_stop = 1+a_loc, 1+a_loc+self.dims["length"]
            a_ix = names.index(a_type)
            a_start, a_stop = a_ix*3, a_ix*3+3
            X_slice = X_set[:, i_start:i_stop, a_start:a_stop]
            return X_slice

        with tf.variable_scope("CartesianToInternal"):
            X_pad = tf.pad(X_cart, [[0, 0], [1, 1], [0, 0]], "CONSTANT")
            X_internal_set = {}
            for atom_ix, name in enumerate(names):
                with tf.variable_scope(name):
                    mode, atoms = dependencies[name]
                    atom_slices = [_slice_atoms(X_pad, t, l) for t, l in atoms]
                    if mode == "backbone":
                        L, T, P = _internals_backbone(*atom_slices)
                    else:
                        L, T, P = _internals_side(*atom_slices)
                    X_internal_set[name] = (L, T, P)

        return X_internal_set

    def _score_logp_internal_coordinates(self, X_model, X_data):
        """ Compute the log probability of the observed coordinates (X_data)
            given the probability of the model (X_model)

            Bonding geometry:
                            |         4.SC         | 
                            |          |           | 
                        ... | - 0.N - 1.Ca - 2.C - | ...
                            |                 |    | 
                            |                3.O   |

            Bond Length distributions are log-normal

            Angular distributions
            N  - Von Mises x Harmonic (Ca-C Dihedral [psi] is flexible)
            Ca - Von Mises-Fisher
            C  - Von Mises x Harmonic ( N-Ca Dihedral [phi] is flexible)
            O  - Von Mises-Fisher
            SC - Von Mises-Fisher
        """

        def _get_scale_param(name, init):
            with tf.variable_scope("Scale_" + name):
                scale = tf.get_variable(
                    name, (), initializer=tf.constant_initializer(init)
                )
                scale_constrained = tf.nn.softplus(scale)
            return scale_constrained

        def _logp_l1(L_model, L_data, mask):
            """  L1 distances """
            with tf.variable_scope("BondLength"):
                residual = L_model - L_data
                logp = -tf.abs(residual)
                logp = tf.reduce_sum(mask * logp) / tf.reduce_sum(mask)
            return logp

        def _logp_normal(L_model, L_data, sigma, mask):
            """  Gaussian distances """
            with tf.variable_scope("BondLength"):
                residual = L_model - L_data
                logp = -tf.square(residual) / (2. * tf.square(sigma)) \
                    - tf.log(sigma) - np.log(np.sqrt(2. * np.pi))
                logp = tf.reduce_sum(mask * logp) / tf.reduce_sum(mask)
            return logp

        def _logp_lognormal(L_model, L_data, sigma, mask):
            """  Gaussian log-distances """
            with tf.variable_scope("BondLength"):
                residual = tf.log(L_model) - tf.log(L_data)
                logp = -tf.square(residual) / (2. * tf.square(sigma)) \
                    - tf.log(sigma) - np.log(np.sqrt(2. * np.pi))
                logp = tf.reduce_sum(mask * logp) / tf.reduce_sum(mask)
            return logp

        def _logp_vonmises_fisher(T_model, T_data, P_model, P_data, kappa,
                                  mask):
            """ Von Mises-Fisher distribution """
            with tf.variable_scope("AnglesVMF"):
                log_2sinh_kappa = kappa + tf.log(1.0 - tf.exp(-kappa))
                logp = kappa * tf.cos(T_model) * tf.cos(T_data) \
                    + tf.sin(T_model) * tf.sin(T_data) \
                    * tf.cos(P_model - P_data) \
                    + tf.log(kappa) - log_2sinh_kappa - np.log(2.0 * np.pi)
                # Average per position
                logp = tf.reduce_sum(mask * logp) / tf.reduce_sum(mask)
            return logp

        def _logp_angles_l1(T_model, T_data, P_model, P_data, mask):
            """ L1 loss for great circle distance """
            with tf.variable_scope("AnglesL1"):
                cosine = tf.cos(T_model) * tf.cos(T_data) \
                    + tf.sin(T_model) * tf.sin(T_data) \
                    * tf.cos(P_model - P_data)
                angle = tf.acos(tf.clip_by_value(cosine, -0.99, 0.99))
                logp = -angle
                # Average per position
                logp = tf.reduce_sum(mask * logp) / tf.reduce_sum(mask)
            return logp

        def _logp_euclidean(T_model, T_data, P_model, P_data, mask):
            """ Featurized Euclidean loss """
            with tf.variable_scope("AnglesEuclidean"):
                # L1 Internal coordinate loss
                dX_model = tf.stack(
                    [tf.cos(T_model), 
                     tf.sin(T_model) * tf.cos(P_model), 
                     tf.sin(T_model) * tf.sin(P_model)], 1
                )
                dX_data = tf.stack(
                    [tf.cos(T_data), 
                     tf.sin(T_data) * tf.cos(P_data), 
                     tf.sin(T_data) * tf.sin(P_data)], 1
                )
                dX_dists = tf.sqrt(
                    tf.reduce_sum(tf.square(dX_data - dX_model), 1) + 1E-5
                )
                # Average per position
                logp = -tf.reduce_sum(mask * dX_dists) / tf.reduce_sum(mask)
            return logp

        def _stack_LTP(LTP_set, names):
            with tf.variable_scope("StackInternals"):
                LTP_stack = [LTP_set[name][0] for name in names] \
                    + [LTP_set[name][1] for name in names]       \
                    + [LTP_set[name][2] for name in names]
                LTP_stack = tf.stack(LTP_stack)
                # Batch, Channels, Length
                LTP_stack = tf.transpose(LTP_stack, [1, 0, 2])
            return LTP_stack

        mask = self.masks["structure_internals"]
        names = ["N", "Ca", "C", "O", "SC"]
        with tf.variable_scope("LogPInternal"):
            LTP_set_model = self._score_full_cartesian_to_internal(X_model)
            LTP_set_data = self._score_full_cartesian_to_internal(X_data)
            logp_LTP = 0.
            for i, name in enumerate(names):
                with tf.variable_scope(name):
                    L_model, T_model, P_model = LTP_set_model[name]
                    L_data, T_data, P_data = LTP_set_data[name]

                    logp_L = _logp_l1(L_model, L_data, mask)
                    kappa_TP = 1.
                    logp_TP = _logp_euclidean(
                        T_model, T_data, P_model, P_data, mask
                    )
                    # Accumulate loss
                    logp_LTP += logp_TP  + logp_L
            logp_LTP /= (3.0 * self.dims["atoms"])

            with tf.variable_scope("Summary"):
                image_mask = tf.expand_dims(tf.expand_dims(mask, 1), 3)
                image_model = tf.expand_dims(
                    _stack_LTP(LTP_set_model, names), 3
                )
                image_data = tf.expand_dims(
                    _stack_LTP(LTP_set_data, names), 3
                )
                tf.summary.image("LTPModel", image_mask * image_model,
                                 collections=["gen_dense"])
                tf.summary.image("LTPData", image_mask * image_data,
                                 collections=["gen_dense"])

            tf.summary.scalar("LogP", tf.squeeze(logp_LTP),
                              collections=["gen_batch", "gen_dense"])
        return logp_LTP

    def _score_squared_dists(self, X):
        """Computes a batch of N distance matrices of size (N,L,L) from a batch
            of N coordinate sets (N,3,L)

            Use ||U-V|| = ||U|| + ||V|| - 2 U.V where ||U|| is the squared
            Euclidean norm
        """
        with tf.variable_scope("Distances"):
            norm = tf.reduce_sum(tf.square(X), 1, keep_dims=True)
            D = norm + \
                tf.transpose(norm, [0, 2, 1]) - 2 * \
                tf.matmul(X, X, adjoint_a=True)
        return D

    def _score_multichannel_dists(self, D):
        """ Convert an atomic distance matrix [B,5L,5L] to
            a multichannel image [B,L,L,25] 
        """
        with tf.variable_scope("DistancesToChannels"):
            B = self.dims["batch"]
            L = self.dims["length"]
            A = self.dims["atoms"]
            # [B,5L,5L] => [B,5L,L,5] => [B,L,5,5L] => [B,L,5,L,5]
            D = tf.reshape(D, [B, A*L, L, A])
            D = tf.transpose(D, perm=[0, 2, 3, 1])
            D = tf.reshape(D, [B, L, A, L, A])
            # [B,L,5,L,5] => [B,L,L,5,5] => [B,L,L,25]
            D = tf.transpose(D, perm=[0, 1, 3, 2, 4])
            D = tf.reshape(D, [B, L, L, A*A])
        return D

    def _score_logp_distances(self, X_model, X_data):
        """ Computes mean squared log-ratio of current vs target distances """

        def _intra_distances(X):
            d_eps = self.hyperparams["scoring"]["logdists"]["eps"]
            X_split = tf.reshape(X, [self.dims["batch"],
                                     self.dims["length"],
                                     self.dims["atoms"], 3]
                                 )
            X_trans = tf.transpose(X_split, [0, 1, 3, 2])
            residual = tf.expand_dims(X_trans, 3) - tf.expand_dims(X_trans, 4)
            # [B,L,5,5]
            D_squared = tf.reduce_sum(tf.square(residual), 2)
            D = tf.sqrt(D_squared + d_eps)
            return D

        with tf.variable_scope("LogPDistances"):
            # Valid positions are those that are further than d_eps
            d_eps = self.hyperparams["scoring"]["logdists"]["eps"]
            mask_coarse = self.masks["structure_coarse_dists"]
            mask_fine = self.masks["structure_fine_dists"]
            Xflat_data = self._score_flatten_coords(X_data)
            Xflat_model = self._score_flatten_coords(X_model)
            logD_model = 0.5 * \
                tf.log(self._score_squared_dists(Xflat_model) + d_eps)
            logD_data = 0.5 * \
                tf.log(self._score_squared_dists(Xflat_data) + d_eps)
            with tf.variable_scope("LogRatio"):
                logD_model_mc = self._score_multichannel_dists(logD_model)
                logD_data_mc = self._score_multichannel_dists(logD_data)
                resid_channels = tf.exp(logD_model_mc) - tf.exp(logD_data_mc)

                channels = self.dims["atoms"]**2
                # Long distances
                contact_cutoff_D = \
                    self.hyperparams["scoring"]["logdists"]["contact"]
                contact_precision = \
                    self.hyperparams["scoring"]["logdists"]["contact_precision"]
                logD_min = tf.reduce_min(tf.concat(
                    axis = 3, values = [logD_model_mc, logD_data_mc]
                ), 3)
                mask_full = self.masks["structure_coarse_dists"]
                mask_contacts = tf.nn.sigmoid(
                    contact_precision * (np.log(contact_cutoff_D) - logD_min)
                ) * mask_full

                pool_width = self.hyperparams["energy"]["backbone_conv"]["width"]
                error_ij = tf.reduce_sum(tf.square(resid_channels), 3) / channels
                error_ij_pooled = self.layers.pool_2D(error_ij, mask_full, pool_width)
                error_ii_pooled = tf.matrix_diag_part(error_ij_pooled)
                error_ij_pooled = 0.5 * error_ij_pooled + 0.25 * (
                    tf.expand_dims(error_ii_pooled, 1)
                    + tf.expand_dims(error_ii_pooled, 2)
                )
                error_ij_pooled = tf.sqrt(error_ij_pooled + 1E-5)
                logp_per_protein = -tf.reduce_sum(mask_contacts * error_ij_pooled, [1,2]) \
                    / tf.reduce_sum(mask_contacts, [1,2])

            with tf.variable_scope("Intra"):
                # Build intra-mask
                off_diag = 1. - \
                    tf.expand_dims(tf.diag(tf.ones([self.dims["atoms"]])), 0)
                mask_intra = tf.reshape(
                    off_diag, [1, 1, self.dims["atoms"], self.dims["atoms"]]
                ) * tf.reshape(
                    self.masks["structure"],
                    [self.dims["batch"], self.dims["length"], 1, 1]
                )

                intraD_model = _intra_distances(X_model)
                intraD_data = _intra_distances(X_data)
                intraD_residual = intraD_data - intraD_model

                # L1 Loss for distances
                error_i = tf.square(intraD_residual)
                error_i = tf.reduce_sum(error_i, [2,3]) / self.dims["atoms"]**2
                pool_width = self.hyperparams["energy"]["backbone_conv"]["width"]
                mask_1D = self.placeholders["structure_mask"]
                error_i_pooled = self.layers.pool_1D(error_i, mask_1D, pool_width)
                error_i_pooled = tf.sqrt(error_i_pooled + 1E-5)
                logp_intra_per = -tf.reduce_sum(mask_1D * error_i_pooled, 1) \
                    / tf.reduce_sum(mask_1D, 1)
                logp_per_protein += logp_intra_per
                self.tensors["logp_distances"] = logp_per_protein
                logp = tf.reduce_mean(logp_per_protein)

            with tf.variable_scope("Contacts"):
                log_cutoff = np.log(10.)
                logD_min_model = tf.reduce_min(logD_model_mc, 3)
                logD_min_data = tf.reduce_min(logD_data_mc, 3)
                ones = tf.ones_like(logD_min_model)
                zeros = tf.zeros_like(logD_min_model)
                contacts_model = tf.where(
                    logD_min_model < log_cutoff, ones, zeros
                )
                contacts_data = tf.where(
                    logD_min_data < log_cutoff, ones, zeros
                )

                mask_full = self.masks["structure_coarse_dists"]
                tp = tf.reduce_sum(
                    mask_full * contacts_model * contacts_data, [1,2]
                )
                fp = tf.reduce_sum(
                    mask_full * contacts_model * (1. - contacts_data), [1,2]
                )
                fn = tf.reduce_sum(
                    mask_full * (1. - contacts_model) * contacts_data, [1,2]
                )
                self.tensors["contact_precision"] = tp / (tp + fp + 1.E-5)
                self.tensors["contact_recall"] = tp / (tp + fn + 1.E-5)

                # Get precision and Specificity
            with tf.variable_scope("Summaries"):
                # Per-batch summaries
                tf.summary.scalar("LogP", logp,
                                  collections=["gen_batch", "gen_dense"])
                tf.summary.histogram("LogP", logp_per_protein,
                                  collections=["gen_batch", "gen_dense"])

                # Dense summaries
                image_model = tf.expand_dims(tf.exp(logD_model), 3)
                image_data = tf.expand_dims(mask_fine * tf.exp(logD_data), 3)
                tf.summary.image(
                    "DistanceModel", image_model, collections=["gen_dense"]
                )
                tf.summary.image(
                    "DistanceTarget", image_data, collections=["gen_dense"]
                )
                # Valid set summaries
                for code in self.hyperparams["test_keys"]:
                    tf.summary.scalar("LogPValid" + code, logp,
                                      collections=[code])
                    tf.summary.histogram("LogPValid" + code, logp_per_protein,
                                      collections=[code])
                    tf.summary.image("ModelValid" + code, image_model,
                                     collections=[code])
                    tf.summary.image("TargetValid" + code, image_data,
                                     collections=[code])
        return logp

    def _score_logp_hbonds(self, X_model, X_data):
        """ Compute the log probability of the hydrogen bonds in the data given
            the model using the defintion of DSSP (Kabsch and Sander, 1983).
            That is, assign a hydrogen bond if

                0.084 * 332 kcals/mol * (1/R_ON + 1/R_CH - 1/R_OH - 1/R_CN)

            is less than 0.5 kcals/mol.
        """
        names = ["N", "Ca", "C", "O", "SC"]
        constants = self.hyperparams["folding"]["constants"]
        _dynamics_eps = constants["xz_eps"]

        def _safe_magnitude(dX):
            """ Magnitudes of [B,3,L] vectors """
            dX_square = tf.reduce_sum(tf.square(dX), axis=1, keep_dims=True)
            D = tf.sqrt(dX_square + _dynamics_eps)
            return D

        def _unit_vec(dX):
            """ Normalize the vector dX """
            D = _safe_magnitude(dX)
            u = dX / D
            return u

        def _inv_distance(X_i, X_j):
            """ Inverse distances for [B,3,L] coord sets """
            with tf.variable_scope("InverseDist"):
                dX = tf.expand_dims(X_i, 2) - tf.expand_dims(X_j, 3)
                d_square = tf.reduce_sum(tf.square(dX), axis=1)
                r = 1. / tf.sqrt(d_square + _dynamics_eps)
            return r

        def _hbond_energies(X):
            """ Compute the all vs all hbond energies """
            # [B,L,15] => [B,L,5,3] => [B,5,3,L]
            with tf.variable_scope("HBondEnergy"):
                X_typed = tf.reshape(X,
                                     [self.dims["batch"], self.dims["length"],
                                      self.dims["atoms"], 3]
                                     )
                X_typed = tf.transpose(X_typed, [0, 2, 3, 1])
                # Build atom sets [B,3,L]
                X_a = {
                    name: X_typed[:, ix, :, :] for ix, name in enumerate(names)
                }
                # Hydrogen built from N(j), Ca(j), and C(j-1)
                X_a["C_prev"] = tf.pad(
                    X_a["C"][:, :, 1:], [[0, 0], [0, 0], [0, 1]], "CONSTANT"
                )
                # Place 1 angstrom away from nitrogen in plane
                X_a["H"] = X_a["N"] + _unit_vec(
                    _unit_vec(X_a["N"] - X_a["C_prev"])
                    + _unit_vec(X_a["N"] - X_a["Ca"])
                )

                # Hydrogen bond energy is relative to DSSP cutoff of 0.5
                energies = 0.5 + (0.084 * 332) * (
                    _inv_distance(X_a["O"], X_a["N"])
                    + _inv_distance(X_a["C"], X_a["H"])
                    - _inv_distance(X_a["O"], X_a["H"])
                    - _inv_distance(X_a["C"], X_a["N"])
                )
            return energies

        with tf.variable_scope("LogPHBonds_lrA"):
            mask = self.masks["structure_coarse_dists"]
            energies_data = _hbond_energies(X_data)
            energies_model = _hbond_energies(X_model)

            # Compute distribution given model
            scale = tf.nn.softplus(tf.get_variable(
                "scale", (), initializer=tf.constant_initializer(1.0)
            ))
            c = tf.nn.softplus(tf.get_variable(
                "coeff", (), initializer=tf.constant_initializer(0.01)
            ))
            b = tf.get_variable(
                "bias", (), initializer=tf.constant_initializer(-1.)
            )
            hbond_logits = c * tf.nn.sigmoid(-scale * energies_model) + b
            hbond_probs = tf.nn.sigmoid(hbond_logits)

            # Assign hbonds in the data
            h_shape = tf.shape(energies_model)
            hbonds_data = tf.where(
                energies_data < 0, tf.ones(h_shape), tf.zeros(h_shape)
            )

            # Probability
            cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=hbonds_data, logits=hbond_logits
            )

            # Build mask of where both i and i-1 are present
            mask_struct = self.masks["structure"]
            mask_struct = mask_struct * tf.pad(
                mask_struct[:, :self.dims["length"]-1], [[0, 0], [1, 0]],
                "CONSTANT"
            )
            ij_dists = self.masks["ij_dist"]
            off_diag = tf.where(
                ij_dists > 2, tf.ones_like(ij_dists), tf.zeros_like(ij_dists)
            )
            mask = tf.expand_dims(mask_struct, 1) \
                * tf.expand_dims(mask_struct, 2) * off_diag
            logp = -(tf.reduce_sum(mask * cross_ent) / tf.reduce_sum(mask))
            with tf.variable_scope("Summaries"):
                # Summaries
                tf.summary.scalar("LogP", logp,
                                  collections=["gen_dense"])
                tf.summary.image("Data",
                                 tf.expand_dims(mask * hbonds_data, 3),
                                 collections=["gen_dense"])
                tf.summary.image("Model",
                                 tf.expand_dims(mask * hbond_probs, 3),
                                 collections=["gen_dense"])
        return logp

    def _score_tm(self, X_model, X_data):
        """ Compute TM scores by optimal superimposition """
        # grad_clip = 0.1
        eps_X = 10.0
        eps_angle = 1.0
        num_steps = 100
        with tf.variable_scope("TMScore"):
            def _coarse_slice(X, atom_ix=1):
                """ Slice representative coordinates C-alpha = 1
                    with output [B,3,L]
                """
                offset = 3 * atom_ix
                X = tf.slice(X, [0, 0, offset], [-1, -1, 3])
                X = tf.transpose(X, [0, 2, 1])
                return X

            mask = self.masks["structure"]
            mask_coords = tf.expand_dims(mask , 1)

            # Compute pre-computables
            X_model = _coarse_slice(X_model)
            X_data = _coarse_slice(X_data)
            av_X_model = tf.reduce_sum(mask_coords * X_model, 2, keep_dims=True) \
                / tf.reduce_sum(mask_coords, 2, keep_dims=True)
            av_X_data = tf.reduce_sum(mask_coords * X_data, 2, keep_dims=True) \
                / tf.reduce_sum(mask_coords, 2, keep_dims=True)
            X_model_centered = X_model - av_X_model
            lengths_float = tf.to_float(self.placeholders["lengths"])
            D0_sq = tf.square(
                1.24 * tf.pow(
                    tf.maximum(lengths_float - 15., tf.ones_like(1.)), 0.33
                    ) - 1.8
            )
            D0_sq = tf.expand_dims(D0_sq, 1)
            total_time = float(num_steps)

            def _transform_unpack(transform):
                """ Unpack translation and rotation angles """
                dX = tf.expand_dims(transform[:, 0:3], 2)
                alpha = transform[:, 3]
                beta = transform[:, 4]
                gamma = transform[:, 5]
                ca, cb, cg = tf.cos(alpha), tf.cos(beta), tf.cos(gamma)
                sa, sb, sg = tf.sin(alpha), tf.sin(beta), tf.sin(gamma)
                R = tf.stack([cb*cg, sa*sb*cg-ca*sg, ca*sb*cg+sa*sg,
                              cb*sg, sa*sb*sg+ca*cg, ca*sb*sg-sa*cg,
                               -sb,           sa*cb,          ca*cb], 1)
                R = tf.reshape(R, [self.dims["batch"], 3, 3])
                return dX, R

            def _tm_scores(transform):
                """ Compute Template Modeling scores (TM-score) between 
                    each X_data and X_model given the transform
                        X_model <- R(X_model - <X_model>) + <X_data> + dX
                    where dX is a translation vector and
                    R is a rotation matrix parameterized
                    by alpha, beta, gamma
                """
                dX, R = _transform_unpack(transform)
                X_transformed = tf.matmul(R, X_model_centered) + av_X_data + dX
                # Compute the distances to the target
                Di_sq = tf.reduce_sum(tf.square(X_transformed - X_data), 1)
                Di_ratio_sq = Di_sq / D0_sq
                tm_site_scores = mask * 1. / (1. + Di_ratio_sq)
                tm_scores = tf.reduce_sum(tm_site_scores, 1) / lengths_float
                return tm_scores

            def _ascend_tm_score(transform, score, t):
                tm_scores = _tm_scores(transform)
                # Soft gradient clipping
                grad = tf.gradients(tf.reduce_sum(tm_scores), transform)[0]
                # Anneal the learning rate
                t_frac = tf.to_float(t) / total_time
                eps = (1. - t_frac) * tf.reshape(
                    [eps_X] * 3 + [eps_angle] * 3,
                    [1, 6]
                )
                # Gradient ascent with sign of gradient
                transform = transform + eps * (
                    tf.sign(grad)
                    + (1. - t_frac) * tf.random_normal(tf.shape(transform))
                )
                return transform, tm_scores, t + 1

            transform_init = tf.zeros([self.dims["batch"], 6])
            score_init = tf.zeros([self.dims["batch"]])
            transform_opt, tm_scores_opt, _ = tf.while_loop(
                lambda transform, score, t: t < num_steps, _ascend_tm_score, 
                [transform_init, score_init, 0]
            )
            self.tensors["tm_scores"] = tm_scores_opt
            tm_score_mean = tf.reduce_mean(tm_scores_opt)
            tf.summary.scalar(
                "TMtrain", tm_score_mean,
                collections=["gen_batch", "gen_dense"]
            )
            tf.summary.histogram(
                "TMtrain", tm_scores_opt,
                collections=["gen_dense"]
            )
            # Valid set summaries
            for code in self.hyperparams["test_keys"]:
                tf.summary.scalar(
                    "TMValid" + code, tm_score_mean, collections=[code]
                )
                tf.summary.histogram(
                    "TMValid" + code, tm_scores_opt, collections=[code]
                )

        with tf.variable_scope("Superimpose"):
            # Apply transform to coordinates
            dX, R = _transform_unpack(transform_opt)
            with tf.variable_scope("Coarse"):
                self.tensors["coordinates_coarse"] = tf.matmul(
                    R, self.tensors["coordinates_coarse"] - av_X_model
                ) + av_X_data + dX
            with tf.variable_scope("Atoms"):
                X_atoms = self.tensors["coordinates_fine"]
                length = self.dims["atoms"] * self.dims["length"]
                # [B,L,15] => [B,5L,3] => [B,3,5L]
                X_atoms = tf.reshape(X_atoms, [self.dims["batch"], length, 3])
                X_atoms = tf.transpose(X_atoms, [0, 2, 1])
                X_atoms = tf.matmul(
                    R, X_atoms - av_X_model
                ) + av_X_data + dX
                # [B,3,5L] => [B,5L,3] => [B,L,15]
                X_atoms = tf.transpose(X_atoms, [0, 2, 1])
                X_atoms = tf.reshape(
                    X_atoms, tf.shape(self.tensors["coordinates_fine"])
                )
                self.tensors["coordinates_fine"] = X_atoms
            with tf.variable_scope("Trajectory"):
                # [B,T,3,L] => [B,3,T,L] => [B,3,TL]
                X_T = self.tensors["trajectory"]
                time_steps = tf.shape(X_T)[1]
                X_T = tf.transpose(X_T, [0,2,1,3])
                length = time_steps * self.dims["length"]
                X_T = tf.reshape(X_T, [self.dims["batch"], 3, length])
                X_T = tf.matmul(
                    R, X_T - av_X_model
                ) + av_X_data + dX
                # [B,3,TL] => [B,3,T,L] => [B,T,3,L]
                X_T = tf.reshape(
                    X_T, 
                    [self.dims["batch"], 3, time_steps, self.dims["length"]]
                )
                X_T = tf.transpose(X_T, [0,2,1,3])
                self.tensors["trajectory"] = X_T
        return tm_score_mean