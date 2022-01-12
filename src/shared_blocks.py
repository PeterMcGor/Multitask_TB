#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue June 12 15:35:14 2019


@author: Pedro M. Gordaliza
"""
import tensorflow as tf
import numpy as np
from VNet import convolution_block, VNet
from utils import IMAGE_CLEF_KEYS

from selu_utils import dropout_selu


def fully_connected_block(x, units=[4096, 2048, 1024, 256], activation=tf.nn.relu6,
                          mode=None, scope_name="mlp_metadata"):
    training = mode == tf.estimator.ModeKeys.TRAIN
    print('TRAINING MODE FULLY', training)
    selu_rates = [0.1] * len(units)
    with tf.variable_scope(scope_name):
        x = tf.layers.flatten(x)
        for i, unit in enumerate(units):
            name = 'Fully_' + str(unit)
            x = tf.layers.dense(x, units=unit, name=name, activation=activation,
                                kernel_initializer=tf.contrib.layers.xavier_initializer())
            x = dropout_selu(x, rate=selu_rates[i], training=training) if activation == tf.nn.selu else x

            x_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=tf.get_default_graph().get_name_scope() + '/' + name)

            if len(x_vars) > 1:
                tf.summary.histogram(name + "_weights", x_vars[0])
                tf.summary.scalar(name + "_mean_weights", tf.reduce_mean(x_vars[0]))
                tf.summary.histogram(name + "_bias", x_vars[1])
        return x


def cnn_shared_features(x, activation=tf.nn.relu6, dropout_rate=[0.1, 0.1, 0.1, 0.1],
                        mode=None, num_of_filters=[256, 128, 64, 32], num_of_convolutions=[3, 3, 3, 3]):
    assert len(num_of_convolutions) == len(num_of_filters)
    training = mode == tf.estimator.ModeKeys.TRAIN
    for i, n_filters in enumerate(num_of_filters):
        for convolution in range(num_of_convolutions[i]):
            x = tf.layers.conv3d(x, n_filters, 1, padding="same")
            x = tf.layers.batch_normalization(x, training=training, renorm=True)
            x = activation(x)
            x = tf.layers.dropout(x, rate=dropout_rate[i], training=training)
    return tf.layers.flatten(x)


def specific_cnn_by_manifestation(x, manifestations=None, activation=tf.nn.relu6, input_size=[8, 8, 8], mode=None):
    training = mode == tf.estimator.ModeKeys.TRAIN
    shape_x = x.get_shape()
    levels = np.floor(np.log2(np.max(input_size))).astype(np.int32)
    last_kernel = np.array([1, 1, 1])
    last_kernel[np.argwhere(input_size == np.amax(input_size))] = 2
    manifestations_features = {}
    manifestations = IMAGE_CLEF_KEYS.keys_as_list() if manifestations is None else manifestations
    for manifestation in manifestations:
        manifestation_x = x
        with tf.variable_scope('down_convolution'):
            for l in range(levels):
                with tf.variable_scope(manifestation + '/encoder/level_' + str(l + 1)):
                    manifestation_x = convolution_block(manifestation_x, 3, 1, activation_fn=activation,
                                                        is_training=training, kernel_size=[2, 2, 2])
                    kernel_s = [2, 2, 2] if l < levels - 1 else last_kernel
                    manifestation_x = tf.layers.conv3d(manifestation_x, shape_x[-1], kernel_size=kernel_s, strides=2)
                    manifestation_x = tf.layers.batch_normalization(manifestation_x, training=training, renorm=True)
                    manifestation_x = activation(manifestation_x)
                    print('L', l, manifestation + '/encoder/level_' + str(l + 1), manifestation_x)
        manifestations_features[manifestation] = tf.layers.flatten(manifestation_x)
    return manifestations_features


def vnet_decoder_style(x, v_net_model=None, manifestations=None, task_up_levels=None, mode=None,
                       activation=tf.nn.relu6):
    assert v_net_model is not None, "VNet model is not provided"
    assert isinstance(v_net_model, VNet)
    # x is here just for compatibility, always will be the encoder features of vnet model
    # x = v_net_model.encoder_features if x is None else x
    is_training = tf.estimator.ModeKeys.TRAIN == mode
    manifestations = IMAGE_CLEF_KEYS.keys_as_list() if manifestations is None else manifestations
    # keep_prob = keep_prob if is_training else 1.0

    manifestations_features = {}
    task_up_levels = 2
    print('ENCODER Features', v_net_model.encoder_features)
    for manifestation in manifestations:
        manifestation_x = v_net_model.decoder_task(mode, task_name=manifestation, task_up_levels=task_up_levels)
        # a capón
        with tf.variable_scope(manifestation + '_features'):
            print(manifestation + 'get_features_in', manifestation_x)
            manifestation_x = tf.layers.conv3d(manifestation_x, 128, [8, 8, 8], strides=[8, 8, 8]) #TODO weigth init here
            manifestation_x = tf.layers.batch_normalization(manifestation_x, training=is_training, renorm=True) # TODO always batch norm??
            manifestation_x = activation(manifestation_x)
            print(manifestation + 'get_features_middle', manifestation_x)

            manifestation_x = tf.layers.conv3d(manifestation_x, 256, [4, 4, 4], strides=1)
            manifestation_x = tf.layers.batch_normalization(manifestation_x, training=is_training, renorm=True)
            manifestation_x = activation(manifestation_x)
            manifestations_features[manifestation] = tf.layers.flatten(manifestation_x)
            print(manifestation + 'get_features_out', manifestation_x)
    return manifestations_features
