import tensorflow as tf
import numpy as np

class Layers:
    def __init__(self, placeholders):
        self.loss = 0.
        self.train_ops = []
        self.placeholders = placeholders
        return

    def nonlinearity(self, x):
      return tf.nn.relu(x)

    def normalize_layer(self, X, mask, name="", anneal_steps=40000,
        adapt_steps=100):
        """ Batch Renormalization """
        with tf.variable_scope("NormalizeLayer" + name):
            # Batch normalization
            C = X.get_shape().as_list()[3]
            mu = tf.get_variable(
                "mu_lrA", shape=(1,1,1,C),
                initializer=tf.zeros_initializer(),
                trainable=False
            )
            sigma = tf.get_variable(
                "sigma_lrA", shape=(1,1,1,C),
                initializer=tf.ones_initializer(),
                trainable=False
            )
            # Compute masked averages
            X = mask * X
            N = tf.reduce_sum(mask, [0,1,2], keep_dims=True)
            av_X = tf.reduce_sum(X, [0,1,2], keep_dims=True) / N
            sqdev_X = tf.square(X - av_X)
            var_X = tf.reduce_sum(mask * sqdev_X, [0,1,2], keep_dims=True) / N
            sigma_X = tf.sqrt(var_X + 1E-5)

            # Batch Renormalization
            step = self.placeholders["global_step"]
            T = tf.clip_by_value(tf.to_float(step) / float(anneal_steps), 0., 1.)
            r_max = 1. + 4 * T 
            d_max = T * 10.
            r = sigma_X / sigma
            d = (av_X - mu) / sigma
            r = tf.stop_gradient(tf.cond(
                step < anneal_steps,
                lambda: tf.clip_by_value(r, 1./ r_max, r_max),
                lambda: r
            ))
            d = tf.stop_gradient(tf.cond(
                step < anneal_steps,
                lambda: tf.clip_by_value(d, -d_max, d_max),
                lambda: d
            ))

            X_train = (X - av_X) / sigma_X * r + d
            X_inference = (X - mu) / sigma
            # Initially use batch
            X_inference = tf.cond(
                step < adapt_steps,
                lambda: X_train,
                lambda: X_inference
            )
            X = mask * tf.cond(
                self.placeholders["training"],
                lambda: X_train,
                lambda: X_inference
            )
            # Moving updates
            beta = tf.cond(step > 100, lambda: 0.99, lambda: 0.)
            mu_update = beta * mu + (1-beta) * av_X
            sigma_update = beta * sigma + (1-beta) * sigma_X
            mu_op = mu.assign(mu_update)
            sigma_op = sigma.assign(sigma_update)
            self.train_ops += [mu_op, sigma_op]
        return X

    def pool_1D(self, X, mask, width):
      """ Pools [B,L] with masking """
      with tf.variable_scope("Pool1DMasked"):
        shape = tf.shape(X)
        X_expand = tf.reshape(X * mask, [shape[0], 1, shape[1], 1])
        X_avg = tf.nn.avg_pool(
            value=X_expand, 
            ksize=[1, 1, width, 1], 
            strides=[1, 1, 1, 1], 
            padding="SAME"
        )
        X_avg = tf.reshape(X_avg, shape)
        return X_avg

    def pool_2D(self, X, mask, width):
      """ Pools [B,L,L] with masking """
      with tf.variable_scope("Pool1DMasked"):
        shape = tf.shape(X)
        X_expand = tf.expand_dims(X * mask, 3)
        X_avg = tf.nn.avg_pool(
            value=X_expand, 
            ksize=[1, width, width, 1], 
            strides=[1, 1, 1, 1], 
            padding="SAME"
        )
        X_avg = tf.squeeze(X_avg, 3)
        return X_avg


    def conv2D(self, inputs, filters, kernel_size=(1, 1),
               padding="same", activation=None,
               reuse=False, name="conv2D", strides=(1, 1),
               dilation_rate=(1, 1), use_bias=True, mask=None,
               batchnorm=False
               ):
        """ 2D convolution layer with weight normalization """

        with tf.variable_scope(name, reuse=reuse):
            # Ensure argument types
            strides = list(strides)
            dilation_rate = list(dilation_rate)
            kernel_size = list(kernel_size)
            padding = padding.upper()
            in_channels = int(inputs.get_shape()[-1])
            out_channels = filters

            # Initialize parameters
            with tf.device('/cpu:0'):
                W_initializer = tf.orthogonal_initializer()
                W_shape = kernel_size + [in_channels, out_channels]
                W = tf.get_variable(
                    "W", shape=W_shape, dtype=tf.float32,
                    initializer=W_initializer, trainable=True
                )

            with tf.device('/cpu:0'):
                g_init = np.ones((out_channels))
                g = tf.get_variable(
                    "g", shape=[out_channels], dtype=tf.float32,
                    initializer=tf.constant_initializer(g_init),
                    trainable=True
                )
                if use_bias and not batchnorm:
                    b = tf.get_variable(
                        "b", shape=[out_channels], dtype=tf.float32,
                        initializer=tf.constant_initializer(np.zeros((out_channels))),
                        trainable=True
                    )

            # Convolution operation
            W_normed = tf.nn.l2_normalize(W, [0, 1, 2])
            h = tf.nn.convolution(
                inputs, filter=W_normed, strides=strides,
                padding=padding, dilation_rate=dilation_rate
            )
            shape = [1, 1, 1, out_channels]
            h = h * tf.reshape(g, shape)
            if use_bias and not batchnorm:
                h = h + tf.reshape(b, shape)
            
            if batchnorm and mask is not None:
                shape = [1, 1, 1, out_channels]
                h = self.normalize_layer(h, mask)
                # Rescale
                gamma = tf.get_variable(
                    "gain", shape=shape, dtype=tf.float32,
                    initializer=tf.constant_initializer(np.ones(shape)),
                    trainable=True
                )
                beta = tf.get_variable(
                    "bias", shape=shape, dtype=tf.float32,
                    initializer=tf.constant_initializer(np.zeros(shape)),
                    trainable=True
                )
                h = gamma * h + beta

            if activation is not None:
                h = activation(h)
        return h


    def conv2D_transpose(self, inputs, filters, kernel_size=(1, 1),
                         padding="same", activation=None,
                         reuse=False, name="conv2D", strides=(1, 1),
                         dilation_rate=(1, 1), use_bias=True, mask=None
                         ):
        """ 2D convolution layer with weight normalization """

        with tf.variable_scope(name, reuse=reuse):
            # Ensure argument types
            strides = list(strides)
            dilation_rate = list(dilation_rate)
            kernel_size = list(kernel_size)
            padding = padding.upper()
            in_channels = int(inputs.get_shape()[-1])
            out_channels = filters

            # Initialize parameters
            with tf.device('/cpu:0'):
                W_initializer = tf.orthogonal_initializer()
                W_shape = kernel_size + [out_channels, in_channels]
                W = tf.get_variable(
                    "W", shape=W_shape, dtype=tf.float32,
                    initializer=W_initializer, trainable=True
                )

            with tf.device('/cpu:0'):
                g_init = np.ones((out_channels))
                g = tf.get_variable(
                    "g", shape=[out_channels], dtype=tf.float32,
                    initializer=tf.constant_initializer(g_init),
                    trainable=True
                )
                if use_bias:
                    b = tf.get_variable(
                        "b", shape=[out_channels], dtype=tf.float32,
                        initializer=tf.constant_initializer(np.zeros((out_channels))),
                        trainable=True
                    )

            # Convolution operation
            W_normed = tf.nn.l2_normalize(W, [0, 1, 2])
            in_shape = tf.shape(inputs)
            output_shape = tf.stack(
                [in_shape[0], in_shape[1], in_shape[2], out_channels])
            h = tf.nn.conv2d_transpose(
                inputs, output_shape=output_shape,
                filter=W_normed, strides=[1]+strides+[1],
                padding=padding
            )
            shape = [1, 1, 1, out_channels]
            h = h * tf.reshape(g, shape)
            if use_bias:
                h = h + tf.reshape(b, shape)
            if activation is not None:
                h = activation(h)
        return h

    def convnet_1D(self, inputs, inner_channels, mask, widths, dilations,
                   dropout_p=0.5, reuse=None, transpose=False):
        """ Residual dilated 1D conv stack. """


        with tf.variable_scope("ConvNet1D"):
            up_layer = inputs
            for i, (width, dilation) in enumerate(zip(widths, dilations)):
                name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
                if transpose:
                    name += "_Trans"
                    f = self.conv2D_transpose
                else:
                    f = self.conv2D
                with tf.variable_scope(name):
                    up_layer_normed = self.normalize_layer(
                        up_layer, mask, name="A"
                    )
                    delta_layer = self.conv2D(up_layer,
                                         filters=inner_channels,
                                         kernel_size=(1, 1),
                                         padding="same",
                                         activation=self.nonlinearity,
                                         reuse=reuse,
                                         name="Mix1" + str(i),
                                         batchnorm=True,
                                         mask=mask)
                    conv_dict = {
                        "inputs": delta_layer,
                        "filters": inner_channels,
                        "kernel_size": (1, width),
                        "padding": "same",
                        "activation": self.nonlinearity,
                        "reuse": reuse,
                        "name": "Conv2" + str(i)
                    }
                    if dilation is not 1:
                        conv_dict["dilation_rate"] = (1, dilation)
                    if transpose is False:
                        conv_dict["batchnorm"] = True
                        conv_dict["mask"] = mask
                    delta_layer = f(**conv_dict)
                    delta_layer = self.conv2D(delta_layer,
                                         filters=inner_channels,
                                         kernel_size=(1, 1),
                                         padding="same",
                                         activation=self.nonlinearity,
                                         reuse=reuse,
                                         name="Mix3" + str(i),
                                         batchnorm=True,
                                         mask=mask)
                    delta_layer = tf.nn.dropout(delta_layer, dropout_p)
                    # delta_layer = self.normalize_layer(
                    #     delta_layer, mask, name="A"
                    # )
                    up_layer = up_layer + delta_layer
        return up_layer


    def convnet_2D(self, inputs, inner_channels, mask, widths, dilations,
                   dropout_p=0.5, reuse=None, transpose=False):
        """ Residual dilated 2D conv stack. """


        with tf.variable_scope("ConvNet2D"):
            up_layer = inputs
            for i, (width, dilation) in enumerate(zip(widths, dilations)):
                name = "Conv" + str(i) + "_" + str(width) + "x" + str(dilation)
                if transpose:
                    name += "_Trans"
                    f = self.conv2D_transpose
                else:
                    f = self.conv2D
                with tf.variable_scope(name):
                    up_layer_normed = self.normalize_layer(
                        up_layer, mask, name="A"
                    )
                    delta_layer = self.conv2D(
                        up_layer,
                        filters=inner_channels,
                        kernel_size=(1, 1),
                        padding="same",
                        activation=self.nonlinearity,
                        reuse=reuse,
                        name="Mix1",
                        batchnorm=True,
                        mask=mask
                    )
                    conv_dict = {
                        "inputs": delta_layer,
                        "filters": inner_channels,
                        "kernel_size": (width, width),
                        "padding": "same",
                        "activation": self.nonlinearity,
                        "reuse": reuse,
                        "name": "Conv2"
                    }
                    if dilation is not 1:
                        conv_dict["dilation_rate"] = (dilation, dilation)
                    if transpose is False:
                        conv_dict["batchnorm"] = True
                        conv_dict["mask"] = mask
                    delta_layer = f(**conv_dict)
                    delta_layer = self.conv2D(
                        delta_layer,
                        filters=inner_channels,
                        kernel_size=(1, 1),
                        padding="same",
                        activation=self.nonlinearity,
                        reuse=reuse,
                        name="Mix3",
                        batchnorm=True,
                        mask=mask
                    )
                    delta_layer = tf.nn.dropout(delta_layer, dropout_p)
                    # delta_layer = self.normalize_layer(
                    #     delta_layer, mask, name="A"
                    # )
                    up_layer = up_layer + delta_layer
        return up_layer


    def expand_1Dto2D(self, input_1D, channels_1D, channels_2D, mask_1D, mask_2D,
                      dropout_p=0.5, extra_input_2D=[], reuse=None):
        """ Gated expansion of 1D conv stack to a 2D conv stack"""

        with tf.variable_scope("1Dto2D"):
            input_1D = self.normalize_layer(
                input_1D, mask_1D, name="A"
            )
            # B,1,L,Channels_1D
            gate_i = self.conv2D(
                input_1D,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                use_bias=True,
                activation=tf.nn.sigmoid,
                reuse=reuse,
                name="iGate1"
            )
            gate_j = self.conv2D(
                input_1D,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                use_bias=True,
                activation=tf.nn.sigmoid,
                reuse=reuse,
                name="jGate2"
            )
            contribution_i = self.conv2D(
                input_1D,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                activation=None,
                reuse=reuse,
                name="iMix1"
            )
            contribution_j = self.conv2D(
                input_1D,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                activation=None,
                reuse=reuse,
                name="jMix2"
            )
            contribution_i *= mask_1D
            contribution_j *= mask_1D
            gate_i *= mask_1D
            gate_j *= mask_1D
            # 2x[B,1,L,K] => [B,K,1,L] + [B,K,L,1]  => [B,K,L,L]
            ij_sum = tf.transpose(contribution_i, [0, 3, 1, 2]) \
                + tf.transpose(contribution_j, [0, 3, 2, 1])
            # [B,K,L,L] => [B,L,L,K]
            ij_input = tf.transpose(ij_sum, [0, 2, 3, 1])
            # [B,K,1,L] + [B,K,L,1]  => [B,K,L,L] => [B,L,L,K]
            gate_ij = tf.transpose(
                tf.transpose(gate_i, [0, 3, 1, 2])
                * tf.transpose(gate_j, [0, 3, 2, 1]), 
                [0, 2, 3, 1]
            )

            # Concatenate extra input if available
            if len(extra_input_2D) > 0:
                input_set = extra_input_2D + [ij_input]
                extra_input_2D = tf.concat(axis=3, values=extra_input_2D)
                extra_input_2D = self.normalize_layer(
                    extra_input_2D, mask_2D, name="B"
                )
                ij_input = tf.concat(
                  axis=3, values=[extra_input_2D, ij_input]
                )

            ij_input = self.normalize_layer(
                ij_input, mask_2D, name="C"
            )

            activation1 = self.conv2D(
                ij_input,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                activation=self.nonlinearity,
                reuse=reuse,
                name="ConcatMix3",
                batchnorm=True,
                mask=mask_2D
            )
            activation2 = self.conv2D(
                activation1,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                activation=self.nonlinearity,
                reuse=reuse,
                name="Mix4",
                batchnorm=True,
                mask=mask_2D
            )
            # activation2 = self.normalize_layer(
            #     activation2, mask_2D, name="A"
            # )
            activation2 = tf.nn.dropout(activation2, dropout_p)
            gate_layer = self.conv2D(
                ij_input,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                activation=tf.nn.sigmoid,
                reuse=reuse,
                name="Gate5",
                batchnorm=True,
                mask=mask_2D
            )
            out_layer = gate_ij * gate_layer * activation2
        return out_layer


    def reduce_2Dto1D(self, input_1D, input_2D, channels_2D, channels_1D, mask_1D, mask_2D,
                      reuse=None):
        """ Gated reduction of 2D conv stack. to a 1D conv stack """

        with tf.variable_scope("2Dto1D"):
            input_1D = self.normalize_layer(
                input_1D, mask_1D, name="1D"
            )
            input_2D = self.normalize_layer(
                input_2D, mask_2D, name="2D"
            )
            # Channel mixing
            delta_layer = self.conv2D(
                input_2D,
                filters=channels_2D,
                kernel_size=(1, 1),
                padding="same",
                activation=self.nonlinearity,
                reuse=reuse,
                name="Mix1",
                batchnorm=True,
                mask=mask_2D
            )
            # Attention query
            query = self.conv2D(
                input_1D,
                filters=channels_2D + 1,
                kernel_size=(1, 1),
                padding="same",
                activation=None,
                reuse=reuse,
                name="Query2"
            )
            # Produce both temperature and address
            beta = tf.expand_dims(tf.exp(query[:,:,:,-1]), 3)
            query = query[:,:,:,:channels_2D]
            # Masked softmax over the edges
            attention = tf.reduce_sum(
              beta * tf.nn.l2_normalize(input_2D,3) * tf.nn.l2_normalize(query,3),
              axis=3, keep_dims=True
            )
            attention_max = tf.reduce_max(attention, axis=1, keep_dims=True)
            attention_weights = mask_2D * tf.exp(attention - attention_max)
            Z = tf.reduce_sum(attention_weights, axis=1, keep_dims=True)
            attention_weights = attention_weights / (Z + 1E-3)
            # Attention-weighted average of the edges
            delta_layer = attention_weights * delta_layer
            delta_layer = tf.reduce_sum(delta_layer, axis=1, keep_dims=True)
            # Channel mixing
            in_layer = tf.concat(axis=3, values=[input_1D, delta_layer])
            delta_layer = self.normalize_layer(
                delta_layer, mask_1D, name="A"
            )
            out_layer = self.conv2D(
                delta_layer,
                filters=channels_1D,
                kernel_size=(1, 1),
                padding="same",
                activation=self.nonlinearity,
                reuse=reuse,
                name="Mix3",
                batchnorm=True,
                mask=mask_1D
            )
        return out_layer


    def conv_graph(self, state_1D, state_2D, channels_1D, channels_2D, widths_1D,
                   dilations_1D, widths_2D, dilations_2D, mask_1D, mask_2D,
                   reuse=None, dropout_p=0.5, aux_1D=None, aux_2D=None):
        with tf.variable_scope("NodeUpdate"):
            # Update nodes given edges
            update_1D = self.reduce_2Dto1D(
                state_1D, state_2D, channels_2D, channels_1D,
                mask_1D, mask_2D, reuse=reuse
            )
            # Input-aware convolution on nodes
            input_1D = [state_1D, update_1D]
            if aux_1D is not None:
              input_1D += [aux_1D]
            update_1D = tf.concat(axis=3, values=input_1D)
            conv_channels = int(update_1D.get_shape()[-1])
            # Conv, LayerNorm, TransConv
            update_1D = self.conv2D(
              update_1D, filters=channels_1D,
              kernel_size=(1, 1), padding="same",
              activation=self.nonlinearity, reuse=reuse,
              name="DownMix",
              batchnorm=True,
                mask=mask_1D
            )
            update_1D = self.normalize_layer(update_1D, mask_1D, name="A")
            update_1D = self.convnet_1D(
                update_1D, channels_1D, mask_1D,
                widths_1D,
                dilations_1D,
                dropout_p,
                reuse=reuse
            )
            state_1D = state_1D + update_1D
            state_1D = self.normalize_layer(state_1D, mask_1D, name="B")
        with tf.variable_scope("EdgeUpdate"):
            # Update edges given nodes
            update_2D = self.expand_1Dto2D(
                state_1D + update_1D,
                channels_1D, channels_2D,
                mask_1D, mask_2D,
                dropout_p,
                reuse=reuse
            )
            # Optional aux input for convolutions
            input_2D = [state_2D, update_2D]
            if aux_2D is not None:
              input_2D += [aux_2D]
            update_2D = tf.concat(axis=3, values=input_2D)
            conv_channels = int(update_2D.get_shape()[-1])
            # 2D convolution is expensive, so just pre-mix
            update_2D = self.conv2D(
              update_2D, filters=channels_2D,
              kernel_size=(1, 1), padding="same",
              activation=self.nonlinearity, reuse=reuse,
              name="DownMix", batchnorm=True,
              mask=mask_2D
            )
            # Convolution on edges
            update_2D = self.normalize_layer(update_2D, mask_2D, name="A")
            update_2D = self.convnet_2D(
                update_2D, channels_2D, mask_2D,
                widths_2D,
                dilations_2D,
                dropout_p,
                reuse=reuse
            )
            state_2D = state_2D + update_2D
            state_2D = self.normalize_layer(state_2D, mask_2D, name="B")
        return state_1D, state_2D


    def reduce_2DtoVec(self, U, mask, num_hidden, num_steps, reuse=None):
        """ Unrolled attentive GRU for reducing a batch of 2D multichannel 
            images to single vector (a la set2set)
        """
        with tf.variable_scope("Reduction"):
            batch_size = tf.shape(U)[0]
            num_in_channels = U.get_shape().as_list()[3]
            h = tf.zeros([tf.shape(U)[0], num_hidden])
            for i in xrange(num_steps):
                with tf.name_scope("Step" + str(i+1)):
                    reuse_layer = None if reuse is None and i is 0 else True
                    # Emit query
                    q = tf.layers.dense(
                        h, num_in_channels, activation=tf.nn.tanh,
                        name="Query1", reuse=reuse_layer
                    )
                    q = tf.layers.dense(
                        q, num_in_channels, activation=None,
                        name="Query2", reuse=reuse_layer
                    )
                    q = tf.reshape(q, [batch_size, 1, 1, num_in_channels])

                    # Compute attention weights
                    a = tf.reduce_sum(U * q, axis=3, keep_dims=True)
                    a_max = tf.reduce_max(a, axis=[1, 2], keep_dims=True)
                    a_weights = mask * tf.exp(a - a_max)
                    Z = tf.reduce_sum(a_weights, axis=[1, 2], keep_dims=True)
                    a_weights = a_weights / (Z + 1E-3)

                    # Compute attention-weighted result
                    U_avg = tf.reduce_sum(a_weights * U, axis=[1, 2])

                    # GRU update
                    hU = tf.concat(axis=1, values=[h, U_avg])
                    gate1 = tf.layers.dense(
                        hU, num_hidden, activation=tf.nn.sigmoid,
                        name="Gate1", reuse=reuse_layer
                    )
                    gate2 = tf.layers.dense(
                        hU, num_hidden, activation=tf.nn.sigmoid,
                        name="Gate2", reuse=reuse_layer
                    )
                    update = tf.nn.tanh(tf.layers.dense(
                        gate2 * h, num_hidden, activation=None,
                        name="h", reuse=reuse_layer
                    ) + tf.layers.dense(
                        U_avg, num_hidden, activation=None,
                        name="U", reuse=reuse_layer
                    ))
                    h = gate1 * h + (1. - gate1) * update
        return h
