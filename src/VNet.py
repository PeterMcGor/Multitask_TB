"""
Created on Tue Mar 22 12:55:57 2019

VNet with BN based on Miguel Monteiro's

@author: Pedro M. Gordaliza


"""

import tensorflow as tf
from Layers import convolution, down_convolution, up_convolution, get_num_channels

top_scope = tf.get_variable_scope()


def convolution_block(layer_input, num_convolutions, keep_prob, activation_fn, is_training=True, kernel_size=[5, 5, 5],
                      batch_normalization=False, renorm=False, activation_after_add=True):
    """
    The residual block follows the convolution --> BN --> activation scheme defined in the original Resnet paper
    (https://arxiv.org/abs/1512.03385). BN is never performed after adding the input to the output. The original paper
     performs the activation operation after the addition however in some cases is reported a better performance avoid
     the activation. This is set by activation_after_add param

    :param layer_input:
    :param num_convolutions:
    :param keep_prob:
    :param activation_fn:
    :param is_training:
    :param kernel_size:
    :param renorm:
    :param batch_normalization:
    :param activation_after_add:
    :return:
    """
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            # convolution(x, spatial_0, spatial_1, spatial_2, num_channel_input, num_channels_output)
            x = convolution(x, [kernel_size[0], kernel_size[1], kernel_size[2], n_channels, n_channels])
            x = tf.layers.batch_normalization(x, training=is_training, renorm=renorm) if batch_normalization else x
            # check shit
            print("GRAPH_NAME", tf.get_default_graph().get_name_scope())  # SUS MUERTOS
            tf.reduce_mean(x, name='MEAN_BN')
            tf.reduce_min(x, name='MIN_BN')
            tf.reduce_max(x, name='MAX_BN')

            if i == num_convolutions - 1:
                x = x + layer_input
                x = activation_fn(x) if activation_after_add else x
                x = tf.nn.dropout(x, keep_prob)
                return x

            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)
    return x


def convolution_block_2(layer_input, fine_grained_features, num_convolutions, keep_prob, activation_fn, is_training=True,
                        kernel_size=[5, 5, 5], batch_normalization=False, renorm=False, activation_after_add=True):
    x = tf.concat((layer_input, fine_grained_features), axis=-1)
    n_channels = get_num_channels(layer_input)
    if num_convolutions == 1:
        with tf.variable_scope('conv_' + str(1)):
            # take into account that here reduce the number of channels. this way is possible to sum up the result to
            # the input
            x = convolution(x, [kernel_size[0], kernel_size[1], kernel_size[2], n_channels * 2, n_channels])
            x = tf.layers.batch_normalization(x, training=is_training, renorm=renorm) if batch_normalization else x
            x = x + layer_input
            x = activation_fn(x) if activation_after_add else x
            x = tf.nn.dropout(x, keep_prob)
        return x
    # First convolution is special. Reduce the number of channels
    with tf.variable_scope('conv_' + str(1)):
        x = convolution(x, [5, 5, 5, n_channels * 2, n_channels])
        x = tf.layers.batch_normalization(x, training=is_training, renorm=renorm) if batch_normalization else x
        x = activation_fn(x)
        x = tf.nn.dropout(x, keep_prob)

    for i in range(1, num_convolutions):
        with tf.variable_scope('conv_' + str(i+1)):
            x = convolution(x, [5, 5, 5, n_channels, n_channels])
            x = tf.layers.batch_normalization(x, training=is_training, renorm=False) if batch_normalization else x
            if i == num_convolutions - 1:
                x = x + layer_input
                x = activation_fn(x) if activation_after_add else x
                x = tf.nn.dropout(x, keep_prob)

            x = activation_fn(x)
            x = tf.nn.dropout(x, keep_prob)

    return x


class VNet(object):
    def __init__(self,
                 num_classes,
                 keep_prob=1.0,
                 num_channels=16,
                 num_levels=4,
                 num_convolutions=(1, 2, 3, 3),
                 bottom_convolutions=3,
                 activation_fn=tf.nn.relu, 
                 only_encoder=False,
                 batch_normalization=False,
                 renormalization=False,
                 activation_after_add=True):
        """
        Implements VNet architecture https://arxiv.org/abs/1606.04797
        :param num_classes: Number of output classes.
        :param keep_prob: Dropout keep probability, set to 1.0 if not training or if no dropout is desired.
        :param num_channels: The number of output channels in the first level, this will be doubled every level.
        :param num_levels: The number of levels in the network. Default is 4 as in the paper.
        :param num_convolutions: An array with the number of convolutions at each level.
        :param bottom_convolutions: The number of convolutions at the bottom level of the network.
        :param activation_fn: The activation function.
        """
        self.num_classes = num_classes
        self.keep_prob = keep_prob
        self.num_channels = num_channels
        assert num_levels <= len(num_convolutions)
        self.num_levels = num_levels
        self.num_convolutions = num_convolutions[:num_levels] if num_levels < len(num_convolutions) else num_convolutions
        # self.num_convolutions = num_convolutions
        self.bottom_convolutions = bottom_convolutions
        self.activation_fn = activation_fn
        self.only_encoder = only_encoder
        self.encoder_features = None
        self.level_features = {}
        self.batch_normalization = batch_normalization
        self.renorm = renormalization
        self.activation_after_add = activation_after_add

    def network_fn(self, x, mode):
        print('MODE ENCODER', mode)
        is_training = tf.estimator.ModeKeys.TRAIN == mode
        print('TRAINING MODE ENCODER', is_training)
#
        keep_prob = self.keep_prob if is_training else 1.0
        
        x, features = self.encoder_fn(x, is_training)
        # self.encoder_fn(x, is_training)
        # Usually for classification purposes
        if self.only_encoder:
            return x

        encoder_features = tf.identity(x)
        x = self.decoder_task(mode, "Segmentation")

        with tf.variable_scope('vnet/output_layer'):
            logits = convolution(x, [1, 1, 1, self.num_channels, self.num_classes])
        with tf.variable_scope('vnet/output_layer/log_var'):
            log_var_seg = convolution(x, [1, 1, 1, self.num_channels, self.num_classes])

        return logits, encoder_features, log_var_seg

    def encoder_fn(self, x, mode):
        is_training = tf.estimator.ModeKeys.TRAIN == mode
        keep_prob = self.keep_prob if is_training else 1.0
        # if the input has more than 1 channel it has to be expanded because broadcasting only works for 1 input channel
        input_channels = int(x.get_shape()[-1])
        with tf.variable_scope('vnet/input_layer'):
            if input_channels == 1:
                x = tf.tile(x, [1, 1, 1, 1, self.num_channels])
                x = tf.layers.batch_normalization(x, training=is_training, renorm=self.renorm) \
                    if self.batch_normalization else x
            else:
                # just in case exit several channels at the input
                x = convolution(x, [5, 5, 5, input_channels, self.num_channels])
                x = tf.layers.batch_normalization(x, training=is_training, renorm=self.renorm) \
                    if self.batch_normalization else x
                x = self.activation_fn(x)

        # features = list()
        for l in range(self.num_levels):
            print("Encoder at level ", l, x)
            with tf.variable_scope('vnet/encoder/level_' + str(l + 1)):
                x = convolution_block(x, self.num_convolutions[l], keep_prob,
                                      activation_fn=self.activation_fn, is_training=is_training)
                if not self.only_encoder:  # just make sense to keep these values when horizontal connection exists
                    self.level_features['level_' + str(l)] = x
                    # features.append(x)
                with tf.variable_scope('down_convolution'):
                    x = down_convolution(x, factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, training=is_training, renorm=self.renorm) \
                        if self.batch_normalization else x
                    x = self.activation_fn(x)
                    x = tf.nn.dropout(x, keep_prob)
                    print("Encoder at level after down convolution", l, x)

        with tf.variable_scope('vnet/bottom_level'):
            x = convolution_block(x, self.bottom_convolutions, keep_prob,
                                  activation_fn=self.activation_fn, is_training=is_training)
            print("Encoder after bottom level", x)
            self.encoder_features = x
        
        return x, self.level_features

    def decoder_task(self, mode, task_name, task_up_levels=None):
        is_training = tf.estimator.ModeKeys.TRAIN == mode
        keep_prob = self.keep_prob if is_training else 1.0
        task_up_levels = self.num_levels if task_up_levels is None else task_up_levels
        x = self.encoder_features

        for l in [level for level in reversed(range(self.num_levels))][:task_up_levels]:
            print('Level up', l)
            with tf.variable_scope('vnet/decoder/'+task_name+'/level_' + str(l)):
                f = self.level_features['level_' + str(l)]
                with tf.variable_scope('up_convolution'):
                    print('vnet/decoder/level_' + str(l + 1), task_name, '_input', x)
                    x = up_convolution(x, tf.shape(f), factor=2, kernel_size=[2, 2, 2])
                    x = tf.layers.batch_normalization(x, training=is_training, renorm=False) \
                        if self.batch_normalization else x
                    x = self.activation_fn(x)
                    x = tf.nn.dropout(x, keep_prob)
                    print('vnet/decoder/level_' + str(l + 1), task_name, '_output_up', x)

                x = convolution_block_2(x, f, self.num_convolutions[l],
                                        keep_prob, activation_fn=self.activation_fn,
                                        is_training=is_training)
        return x



