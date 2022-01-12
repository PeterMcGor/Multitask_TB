#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:37:18 2019

@author: pedro
"""
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils

import numbers

from Layers import get_num_channels, xavier_initializer_convolution, constant_initializer

def convolutional_selu_block(layer_input, num_convolutions, rate, activation_fn,  is_training = False):
    x = layer_input
    n_channels = get_num_channels(x)
    for i in range(num_convolutions):
        with tf.variable_scope('conv_SELU_'+str(i+1)):
            x = convolution_selu(x, [5, 5, 5, n_channels, n_channels])
            if i == num_convolutions - 1:
                x = x + layer_input
            x = tf.nn.selu(x)
            x = dropout_selu(x, rate, training=is_training)
            tf.summary.histogram('conv_SELU_'+str(i+1), x)
    return x



def convolution_selu(x, filter, padding='SAME', strides=None, dilation_rate=None):
    """
    Same than convolution but different variable initialization
    """
    w = tf.get_variable(name='weights', initializer=xavier_initializer_convolution(shape=filter, dist='normal'))
    b = tf.get_variable(name='biases', initializer=constant_initializer(0, shape=filter[-1]) )

    return tf.nn.convolution(x, w, padding, strides, dilation_rate) + b


def dropout_selu(x, rate, alpha= -1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0,
                 noise_shape=None, seed=None, name=None, training=False):
    """Dropout to a value with rescaling."""

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name="x")
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                                             "range (0, 1], got %g" % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name="keep_prob")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name="alpha")
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

        if tensor_util.constant_value(keep_prob) == 1:
            return x

        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1-binary_tensor)

        a = math_ops.sqrt(fixedPointVar / (keep_prob *((1-keep_prob) * math_ops.pow(alpha-fixedPointMean,2) + fixedPointVar)))

        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret

    with ops.name_scope(name, "dropout", [x]) as name:
        return utils.smart_cond(training,
            lambda: dropout_selu_impl(x, rate, alpha, noise_shape, seed, name),
            lambda: array_ops.identity(x))