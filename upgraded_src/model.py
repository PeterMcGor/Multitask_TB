#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:35:14 2019

In model will be the appropiate wrappers in order to use TF estimators 
within each model

@author: Pedro M. Gordaliza
"""
import sys
import os

import numpy as np

import tensorflow as tf
from tensorflow.python.saved_model import signature_constants

from VNet import VNet
from UNet import UNet
from utils import IMAGE_CLEF_KEYS
import losses
from shared_blocks import fully_connected_block, specific_cnn_by_manifestation


class Aim:
    CLASSIFICATION = "Multi-task_clf"
    SEGMENTATION = "Lung_Seg"
    CLASS_AND_SEG = CLASSIFICATION + "_and_" + SEGMENTATION


class UNCERTANTY_TYPE:
    HOMOCEDASTIC = 'Homoscedastic'
    HETEROCEDASTIC = 'Heterocedastic'
    EPISTEMIC = 'Epistemic'


class INIT_BIAS:
    MINORITY = "Bias_for_minority_class"
    MAJORITY = "Bias_for_majority_class"
    @staticmethod
    def get_bias_dict(bias, pi_prior_bias_init=0.01):
        assert bias == INIT_BIAS.MINORITY or bias == INIT_BIAS.MAJORITY, "Incorrect init bias "+bias
        bias_init = np.log(pi_prior_bias_init / (1 - pi_prior_bias_init))
        if bias == INIT_BIAS.MINORITY:
            return {IMAGE_CLEF_KEYS.SEVERITY: 2.,
                    IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: [bias_init, 0.],
                    IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: [bias_init, 0.],
                    IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: [0., bias_init],
                    IMAGE_CLEF_KEYS.CTR_CALCIFICATION: [0., bias_init],
                    IMAGE_CLEF_KEYS.CTR_PLEURISY: [0., bias_init],
                    IMAGE_CLEF_KEYS.CTR_CAVERNS: [0., bias_init]}
        else:
            return {IMAGE_CLEF_KEYS.SEVERITY: 2.,
                    IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: [0., bias_init],
                    IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: [0., bias_init],
                    IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: [bias_init, 0.],
                    IMAGE_CLEF_KEYS.CTR_CALCIFICATION: [bias_init, 0.],
                    IMAGE_CLEF_KEYS.CTR_PLEURISY: [bias_init, 0.],
                    IMAGE_CLEF_KEYS.CTR_CAVERNS: [bias_init, 0.]}


def custom_sigmoid(x, k=1, a=3.5):
    return 1 / (1 + tf.exp(-k * (x - a)))


def multi_task_block(x, meta_data, n_channels, uncertainty_type=UNCERTANTY_TYPE.HOMOCEDASTIC,
                     mode=None, use_metadata=True, batch_normalization=False, renorm=False,
                     bias_initializer=INIT_BIAS.MINORITY):
    """
    :param renorm:
    :param batch_normalization:
    :param use_metadata:
    :param mode:
    :param n_channels:
    :param meta_data:
    :param x:
    :param uncertainty_type:
    :param bias_initializer:
    :return:
    """

    def dense_block(input_layer, name, layers_units, activations, bias_initializer_last_layer):
        training = mode == tf.estimator.ModeKeys.TRAIN
        print("Dense block", batch_normalization, 'Training', training)
        assert len(layers_units) == len(activations)
        output_layer = input_layer
        for i, layer_i in enumerate(layers_units[:-1]):
            output_layer = tf.compat.v1.layers.dense(output_layer, units=layer_i, name=name + '_' + str(layer_i),
                                           kernel_initializer=tf.compat.v1.keras.initializers.random_normal(stddev=0.01))
            ### TODO Este tiene pinta de ser 0 por como funciona #########
            output_layer = tf.compat.v1.layers.batch_normalization(output_layer, training=training, renorm=renorm) \
                if batch_normalization else output_layer
            print("GRAPH_NAME_2", tf.compat.v1.get_default_graph().get_name_scope())
            tf.reduce_mean(input_tensor=output_layer, name='MEAN_OUT_LAYER')
            #  tf.reduce_mean()
            tf.reduce_min(input_tensor=output_layer, name='MIN_OUT_LAYER')
            tf.reduce_max(input_tensor=output_layer, name='MAX_OUT_LAYER')
            output_layer = activations[i](output_layer)

            output_layer_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=tf.compat.v1.get_default_graph().get_name_scope() + '/'
                                                        + name + '_' + str(layer_i))
            if len(output_layer_vars) > 1:
                tf.compat.v1.summary.histogram(name + '_' + str(layer_i) + "_weights", output_layer_vars[0])
                tf.compat.v1.summary.histogram(name + '_' + str(layer_i) + "_biases", output_layer_vars[1])
                tf.compat.v1.summary.histogram(name + '_' + str(layer_i) + "_activations", output_layer)

        preds = tf.compat.v1.layers.dense(output_layer, units=layers_units[-1], name=name + '_' + str(layers_units[-1]),
                                activation=activations[-1], kernel_initializer=tf.compat.v1.initializers.zeros(),
                                bias_initializer=tf.compat.v1.keras.initializers.constant(bias_initializer_last_layer))

        output_layer_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=tf.compat.v1.get_default_graph().
                                              get_name_scope() + '/' + name + '_' + str(layers_units[-1]))
        if len(output_layer_vars) > 1:
            tf.compat.v1.summary.histogram(name + '_' + str(layers_units[-1]) + "_weights", output_layer_vars[0])
            tf.compat.v1.summary.histogram(name + '_' + str(layers_units[-1]) + "_biases", output_layer_vars[1])
            tf.compat.v1.summary.histogram(name + '_' + str(layers_units[-1]) + "_activations", output_layer)

        return output_layer, preds

    def get_log_var_uncertainty(name, uncertainty, input_layer=None):
        log_var = 0.
        with tf.compat.v1.variable_scope("log_variances"):
            if uncertainty == UNCERTANTY_TYPE.HOMOCEDASTIC:
                suffix = '_log_var_hom'
                log_var = tf.compat.v1.get_variable(name + suffix, shape=(1,), initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"),
                                          trainable=True, dtype=tf.float32)
            elif uncertainty == UNCERTANTY_TYPE.HETEROCEDASTIC:
                suffix = '_log_var_het'
                hetero_units = [256, 128, 64, 32, 16, 1]
                hetero_acts = [tf.nn.relu6] * (len(hetero_units) - 1) + [None]
                log_var = dense_block(input_layer, name + suffix, hetero_units, hetero_acts, 0.)[1]

        return log_var

    # Epistemic has nothing to do here
    unc_id = 'No_Model_Unc'
    if uncertainty_type == UNCERTANTY_TYPE.HOMOCEDASTIC:
        unc_id = 'Homost'
    elif uncertainty_type == UNCERTANTY_TYPE.HETEROCEDASTIC:
        unc_id = 'Heterost'
    dict_check = True if isinstance(x, dict) else False
    n_levels = 4

    manifestations_list = IMAGE_CLEF_KEYS.keys_as_list()
    binary_manifestations_list = IMAGE_CLEF_KEYS.keys_as_list(just_binary=True)
    binary_layers_units = [256, 128, 64, 32, 16, 2] if dict_check else [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 2]
    binary_layers_acts = [tf.nn.relu6] * (len(binary_layers_units) - 1) + [None]
    binary_layers_units_dict = {manifestation: binary_layers_units for manifestation in binary_manifestations_list}
    binary_layers_acts_dict = {manifestation: binary_layers_acts for manifestation in binary_manifestations_list}
    severity_layers_units = [256, 128, 64, 32, 16, 1] if dict_check else [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 1]
    severity_layers_acts = [tf.nn.relu6] * len(severity_layers_units)
    layers_units_dict = {**{IMAGE_CLEF_KEYS.SEVERITY: severity_layers_units}, **binary_layers_units_dict}
    layers_acts_dict = {**{IMAGE_CLEF_KEYS.SEVERITY: severity_layers_acts}, **binary_layers_acts_dict}
    # pi_prior_bias_init = 0.01
    # bias_init = np.log(pi_prior_bias_init / (1 - pi_prior_bias_init))
    # #### bias for minority class
    bias_init_dict = INIT_BIAS.get_bias_dict(bias_initializer)
    # bias_init_dict = {IMAGE_CLEF_KEYS.SEVERITY: 2.,
    #                   IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: [bias_init, 0.],
    #                   IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: [bias_init, 0.],
    #                   IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: [0., bias_init],
    #                   IMAGE_CLEF_KEYS.CTR_CALCIFICATION: [0., bias_init],
    #                   IMAGE_CLEF_KEYS.CTR_PLEURISY: [0., bias_init],
    #                   IMAGE_CLEF_KEYS.CTR_CAVERNS: [0., bias_init]}
    # #### bias for majority class
    # bias_init_dict = {IMAGE_CLEF_KEYS.SEVERITY: 2.,
    #                   IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: [0., bias_init],
    #                   IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: [0., bias_init],
    #                   IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: [bias_init, 0.],
    #                   IMAGE_CLEF_KEYS.CTR_CALCIFICATION: [bias_init, 0.],
    #                   IMAGE_CLEF_KEYS.CTR_PLEURISY: [bias_init, 0.],
    #                   IMAGE_CLEF_KEYS.CTR_CAVERNS: [bias_init, 0.]}

    manifestation_logits = {}
    manifestations_log_vars = {}

    for manifestation in manifestations_list:
        with tf.compat.v1.variable_scope(manifestation + "_features"):
            # print('The_X_at_'+manifestation, x)
            if dict_check:
                manifestation_x = x[manifestation]
                print('Manifestation_x', manifestation, manifestation_x, 'use_metadata', use_metadata)

                if use_metadata:
                    manifestation_x = tf.concat([manifestation_x,
                                                 fully_connected_block(meta_data, units=
                                                 [np.max([16, n_channels * 2 ** n_levels / 4]).astype(np.int32)],
                                                                       mode=mode,  # Todo check this MODE
                                                                       scope_name='mlp_meta_data_' + manifestation)],
                                                axis=1)
            else:
                manifestation_x = x
                if use_metadata:
                    manifestation_x = tf.concat([manifestation_x,
                                                 fully_connected_block(meta_data, units=[128],
                                                                       mode=mode,
                                                                       scope_name='mlp_meta_data_b' + manifestation)],
                                                axis=1)
            print('The_X_at_after_' + manifestation, manifestation_x)

            # TODO Esto va en este scope o fuera?

            manifestation_l, manifestation_logit = dense_block(manifestation_x, manifestation,
                                                               layers_units_dict[manifestation],
                                                               layers_acts_dict[manifestation],
                                                               bias_init_dict[manifestation])
            # TODO x or just the last layer _l
            manifestation_log_var = get_log_var_uncertainty(manifestation, uncertainty_type, manifestation_x)
            manifestation_logits[manifestation] = manifestation_logit
            manifestations_log_vars[manifestation] = manifestation_log_var
            tf.compat.v1.summary.scalar('log_var_' + unc_id + '_' + manifestation, tf.reduce_mean(input_tensor=manifestation_log_var))

    return manifestation_logits, manifestations_log_vars


class Segmentation(object):
    pass


class Model(object):
    VNet = 'VNet'
    UNet = 'UNet'

    def __init__(self, aim, architecture_params, shared_block_params, model_params):
        # Segmentation (Aim.SEGMENTATION), Classification (Aim.CLASSIFICATION) or both (Aim.CLASS_AND_SEG)
        self.aim = aim
        # Model architecture: VNet, Resnet, etc.
        self.architecture = architecture_params['architecture']
        self.arq_params = architecture_params
        self.model = self.__get_model__()
        self.shared_block_params = shared_block_params
        self.shared_fn = shared_block_params.pop('architecture')
        self.training_mode = None
        self.model_params = model_params
        self.model_dir = model_params['model_dir']

    def __get_model__(self):
        if self.architecture == self.VNet:
            return VNet(1, keep_prob=self.arq_params['keep_prob'],
                        num_channels=self.arq_params['num_channels'],
                        num_levels=self.arq_params['num_levels'],
                        activation_fn=self.arq_params['activation'])
        elif self.architecture == self.UNet:
            return UNet(1, n_base_filters=16,
                        depth=self.arq_params['num_levels'],
                        batch_norm=True)
        else:
            print("Incorrect arquitecture", self.architecture)
            sys.exit(1)

    def model_fn(self, features, labels, mode, params):
        def get_segmetation_preds(logits_seg):
            squashed_logits_seg = tf.nn.sigmoid(logits_seg, name="softmax_tensor")

            seg_predictions = {
                "classes": squashed_logits_seg > 0.5,  # tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.sigmoid(logits_seg, name="softmax_tensor")
            }
            return seg_predictions

        def get_logits_preds(logits_clf):
            sever = logits_clf[IMAGE_CLEF_KEYS.SEVERITY]  # logits_clf[:, 0]

            logits_clf = tf.concat([logits_clf[manifestation] for manifestation in
                                    IMAGE_CLEF_KEYS.keys_as_list(just_binary=True)], axis=1, name='Logits_CLF')
            clf_sigmoid = tf.nn.sigmoid(logits_clf, name='SIGMOID_OUT')
            clf_sigmoid_preds = clf_sigmoid > 0.5
            clf_predictions = {
                IMAGE_CLEF_KEYS.SEVERITY: sever,

                IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: clf_sigmoid_preds[:, 0],
                IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: clf_sigmoid_preds[:, 1],
                IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: clf_sigmoid_preds[:, 2],
                IMAGE_CLEF_KEYS.CTR_CALCIFICATION: clf_sigmoid_preds[:, 3],
                IMAGE_CLEF_KEYS.CTR_PLEURISY: clf_sigmoid_preds[:, 4],
                IMAGE_CLEF_KEYS.CTR_CAVERNS: clf_sigmoid_preds[:, 5],
                IMAGE_CLEF_KEYS.SVR_SEVERITY: sever > 3.5
            }
            return clf_predictions

        def get_subnet_estimator(scope, loss_tensor, optz, learning_r, optz_params, step=None):
            variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            # gradients = tf.gradients(loss_tensor, variables)

            # print(scope + '_VARS', variables)

            if len(optz_params) == 0:
                optz = optz(learning_rate=learning_r)
            else:
                new_params = [learning_r] + optz_params
                optz = optz(*new_params)

            # grads_and_vars = optz.compute_gradients(loss_tensor, variables)
            # vars_with_grad = [v for g, v in grads_and_vars if g is not None]
            # train_subnet_operation = optz.apply_gradients(zip(gradients, variables), global_step=global_step)
            # train_subnet_operation = optz.apply_gradients(vars_with_grad, global_step=global_step)
            train_subnet_operation = optz.minimize(loss_tensor, global_step=step if step is not None else None,
                                                   var_list=variables)
            return train_subnet_operation

        def get_segmentation_summaries():
            for batch in range(1):
                max_outputs = 10
                from_i = 20
                to_j = 60
                step = 2
                log_var_log = log_vars_seg[batch, :, :, from_i:to_j:step, :]
                seg_predictions_log = seg_predictions["probabilities"][batch, :, :, from_i:to_j:step, :]
                seg_classes_log = 255 * tf.cast(seg_predictions["classes"][batch, :, :, from_i:to_j:step, :],
                                                dtype=tf.uint8)
                features_predictions_log = features['features']['x'][batch, :, :, from_i:to_j:step, :]
                mask_log = mask[batch, :, :, from_i:to_j:step, :]

                tf.compat.v1.summary.image("image", tf.transpose(a=features_predictions_log, perm=[2, 1, 0, 3]), max_outputs=max_outputs)
                tf.compat.v1.summary.image("uncertainty_seg", tf.transpose(a=log_var_log, perm=[2, 1, 0, 3]), max_outputs=max_outputs)
                tf.compat.v1.summary.image("probs_seg", tf.transpose(a=seg_predictions_log, perm=[2, 1, 0, 3]), max_outputs=max_outputs)
                tf.compat.v1.summary.image("class_seg", tf.transpose(a=seg_classes_log, perm=[2, 1, 0, 3]), max_outputs=max_outputs)
                tf.compat.v1.summary.image("masks", tf.transpose(a=mask_log, perm=[2, 1, 0, 3]), max_outputs=max_outputs)

        # Estimators kind of dark this parameter which is most of the times needed
        self.training_mode = tf.estimator.ModeKeys.TRAIN == mode
        log_vars_clf = None
        log_vars_seg = None
        clf_logits = None
        seg_logits = None
        hierarchy = params['hierarchy']
        tf.strings.strip(mode, name='MODE_NAME')

        x = features['features']['x'] if mode != tf.estimator.ModeKeys.PREDICT else features['x']
        meta_info = features['features']['meta_info'] \
            if mode != tf.estimator.ModeKeys.PREDICT else features['meta_info']
        # ###################             INFERENCE            ###########################
        with tf.compat.v1.variable_scope("cnn_features"):
            if self.aim == Aim.CLASSIFICATION:
                classification_cnn_features = self.model.encoder_fn(x, mode)[0]
            else:
                seg_logits, classification_cnn_features, log_vars_seg = self.model.network_fn(x, mode)

        if self.aim != Aim.CLASSIFICATION:
            seg_predictions = get_segmetation_preds(seg_logits)

        if self.aim != Aim.SEGMENTATION:  # Get classification
            with tf.compat.v1.variable_scope('after_' + self.architecture + '_features'):
                self.shared_block_params['mode'] = mode
                # specific_cnn_by_manifestations block reduce per manifestations shared features from the backbone
                # architecture to a [bs, 1, 1, 1, classification_cnn_features.shape[-1]] (usually 256 features)
                x = self.shared_fn(classification_cnn_features, **self.shared_block_params)  # x is a dict

            with tf.compat.v1.variable_scope("multi_task_features"):
                clf_logits, log_vars_clf = multi_task_block(x, meta_info, self.arq_params['num_channels'],
                                                            uncertainty_type=params['aleatoric_uncertainty'],
                                                            mode=mode, use_metadata=params['employ_meta_info'],
                                                            batch_normalization=params['BN_dense_block'],
                                                            renorm=params['renorm_dense_block'],
                                                            bias_initializer=params['init_bias'])

                # "CLF_logits should be a dict with eah element (bs, 2) LOG_VARS  another list with the number of
                # tasks. (each element) For homo should be (1, ) and for heter (bs,)

            clf_predictions = get_logits_preds(clf_logits)

            # ###################         END INFERENCE            ###########################

        if mode == tf.estimator.ModeKeys.PREDICT:
            if self.aim == Aim.CLASSIFICATION:
                export_outputs = {'clf_prediction': tf.estimator.export.PredictOutput(clf_predictions)}
                return tf.estimator.EstimatorSpec(mode=mode, predictions=clf_predictions, export_outputs=export_outputs)
            elif self.aim == Aim.SEGMENTATION:
                export_outputs = {'seg_prediction': tf.estimator.export.PredictOutput(seg_predictions)}
                return tf.estimator.EstimatorSpec(mode=mode, predictions=seg_predictions, export_outputs=export_outputs)
            else:  # Aim.CLASS_AND_SEG
                export_outputs = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                      tf.estimator.export.PredictOutput({**clf_predictions, **seg_predictions}),
                                  'seg_prediction': tf.estimator.export.PredictOutput(seg_predictions)}

                return tf.estimator.EstimatorSpec(mode=mode, predictions={**clf_predictions, **seg_predictions},
                                                  export_outputs=export_outputs)

        # #############   Loss for TRAIn and EVAL Modes########################################
        labels = labels['labels']
        name = labels['Name']
        mask = labels['Mask']
        labels = labels['y']
        tf.strings.strip(name, name='Check_rand')  # Just to quality check
        loss_fn = params['loss_fn']

        # TODO include as inside the class to provide the proper structure
        loss, av_per_task_loss = loss_fn(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
        loss = tf.clip_by_value(loss, tf.float32.min, tf.float32.max)
        if self.aim != Aim.CLASSIFICATION:
            get_segmentation_summaries()
            eval_metric_ops_seg = {"accuracy": tf.compat.v1.metrics.accuracy(labels=mask, predictions=seg_predictions["classes"]),
                                   "IOU": tf.compat.v1.metrics.mean_iou(labels=mask, predictions=seg_predictions["classes"],
                                                              num_classes=2)}

        # ##########General Evals#####
        if self.aim != Aim.SEGMENTATION:
            binary_sigmoid = {}
            binary_sigmoid_2 = {}
            prediction = {}
            for manifestation in IMAGE_CLEF_KEYS.keys_as_list(just_binary=True):
                softmax = tf.nn.softmax(clf_logits[manifestation], name='Softmax_' + manifestation)
                max = tf.clip_by_value(tf.reduce_max(input_tensor=softmax, axis=1,
                                                     name='Max_' + manifestation), tf.keras.backend.epsilon(),
                                       1 - tf.keras.backend.epsilon())
                arg_max = tf.cast(tf.argmax(input=softmax, axis=1, name='argmax_' + manifestation), max.dtype)
                binary_sigmoid[manifestation] = tf.add(arg_max * max,
                                                       (1 - arg_max) * (1 - max), name='out_prob_' + manifestation)
                prediction[manifestation] = arg_max

                softmax_2 = tf.nn.softmax(clf_logits[manifestation] * tf.exp(-log_vars_clf[manifestation]),
                                          name='Softmax_2' + manifestation)
                max_2 = tf.clip_by_value(tf.reduce_max(input_tensor=softmax_2, axis=1, name='Max_2' + manifestation),
                                         tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
                arg_max_2 = tf.cast(tf.argmax(input=softmax_2, axis=1, name='argmax_2' + manifestation), max_2.dtype)
                binary_sigmoid_2[manifestation] = tf.add(arg_max_2 * max_2, (1 - arg_max_2) * (1 - max_2),
                                                         name='out_prob_2' + manifestation)
            # binary_sigmoid = tf.concat(binary_sigmoid, axis=0, name='softmax_logits')

            ## TODO esto va solo en EVAL o qué???

            AUC_CTR_left_lung_affected = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED],
                                                        binary_sigmoid[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED])
            AUC_CTR_right_lung_affected = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED],
                                                         binary_sigmoid[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED])
            AUC_CTR_lung_capacity_decrease = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE],
                                                            binary_sigmoid[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE])
            AUC_CTR_calcification = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_CALCIFICATION],
                                                   binary_sigmoid[IMAGE_CLEF_KEYS.CTR_CALCIFICATION])
            AUC_CTR_pleurisy = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_PLEURISY],
                                              binary_sigmoid[IMAGE_CLEF_KEYS.CTR_PLEURISY])
            AUC_CTR_caverns = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_CAVERNS],
                                             binary_sigmoid[IMAGE_CLEF_KEYS.CTR_CAVERNS])
            severity_sigmoid = custom_sigmoid(clf_logits[IMAGE_CLEF_KEYS.SEVERITY])
            print("SEVERITY_SIGMOID", severity_sigmoid)
            severity_sigmoid_2 = custom_sigmoid(clf_logits[IMAGE_CLEF_KEYS.SEVERITY]
                                                * tf.exp(-log_vars_clf[IMAGE_CLEF_KEYS.SEVERITY]))
            print("SEVERITY_SIGMOID_2", severity_sigmoid_2)
            AUC_SVR_severity = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.SVR_SEVERITY], severity_sigmoid)

            AUC_CTR_left_lung_affected_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED],
                                                          binary_sigmoid_2[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED])
            AUC_CTR_right_lung_affected_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED],
                                                           binary_sigmoid_2[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED])
            AUC_CTR_lung_capacity_decrease_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE],
                                                              binary_sigmoid_2[
                                                                  IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE])
            AUC_CTR_calcification_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_CALCIFICATION],
                                                     binary_sigmoid_2[IMAGE_CLEF_KEYS.CTR_CALCIFICATION])
            AUC_CTR_pleurisy_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_PLEURISY],
                                                binary_sigmoid_2[IMAGE_CLEF_KEYS.CTR_PLEURISY])
            AUC_CTR_caverns_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.CTR_CAVERNS],
                                               binary_sigmoid_2[IMAGE_CLEF_KEYS.CTR_CAVERNS])
            AUC_SVR_severity_2 = tf.compat.v1.metrics.auc(labels[IMAGE_CLEF_KEYS.SVR_SEVERITY], severity_sigmoid_2)

            eval_metric_ops_clf = {
                "RMSE_" + IMAGE_CLEF_KEYS.SEVERITY: tf.compat.v1.metrics.root_mean_squared_error(
                    labels=tf.cast(labels[IMAGE_CLEF_KEYS.SEVERITY], tf.float32),
                    predictions=clf_logits[IMAGE_CLEF_KEYS.SEVERITY]),
                "AUC_" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: AUC_CTR_left_lung_affected,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: AUC_CTR_right_lung_affected,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: AUC_CTR_lung_capacity_decrease,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: AUC_CTR_calcification,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_PLEURISY: AUC_CTR_pleurisy,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_CAVERNS: AUC_CTR_caverns,
                "AUC_" + IMAGE_CLEF_KEYS.SVR_SEVERITY: AUC_SVR_severity,

                "AUC_2" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: AUC_CTR_left_lung_affected_2,
                "AUC_2" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: AUC_CTR_right_lung_affected_2,
                "AUC_2" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: AUC_CTR_lung_capacity_decrease_2,
                "AUC_2" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: AUC_CTR_calcification_2,
                "AUC_2" + IMAGE_CLEF_KEYS.CTR_PLEURISY: AUC_CTR_pleurisy_2,
                "AUC_2" + IMAGE_CLEF_KEYS.CTR_CAVERNS: AUC_CTR_caverns_2,
                "AUC_2" + IMAGE_CLEF_KEYS.SEVERITY: AUC_SVR_severity_2,

                "ACC_" + IMAGE_CLEF_KEYS.SVR_SEVERITY: tf.compat.v1.metrics.accuracy(labels[IMAGE_CLEF_KEYS.SVR_SEVERITY],
                                                                           severity_sigmoid > 0.5),

                "ACC_" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: tf.compat.v1.metrics.accuracy(
                    labels[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED],
                    prediction[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED]),

                "ACC_" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: tf.compat.v1.metrics.accuracy(
                    labels[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED],
                    prediction[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED]),

                "ACC_" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: tf.compat.v1.metrics.accuracy(
                    labels[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE],
                    prediction[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE]),

                "ACC_" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: tf.compat.v1.metrics.accuracy(
                    labels[IMAGE_CLEF_KEYS.CTR_CALCIFICATION],
                    prediction[IMAGE_CLEF_KEYS.CTR_CALCIFICATION]),

                "ACC_" + IMAGE_CLEF_KEYS.CTR_PLEURISY: tf.compat.v1.metrics.accuracy(
                    labels[IMAGE_CLEF_KEYS.CTR_PLEURISY],
                    prediction[IMAGE_CLEF_KEYS.CTR_PLEURISY]),

                "ACC_" + IMAGE_CLEF_KEYS.CTR_CAVERNS: tf.compat.v1.metrics.accuracy(
                    labels[IMAGE_CLEF_KEYS.CTR_CAVERNS],
                    prediction[IMAGE_CLEF_KEYS.CTR_CAVERNS]),

                "FPE_" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: tf.compat.v1.metrics.false_positives(
                    labels[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED],
                    prediction[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED]),

                "FPE_" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: tf.compat.v1.metrics.false_positives(
                    labels[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED],
                    prediction[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED]),

                "FPE_" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: tf.compat.v1.metrics.false_positives(
                    labels[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE],
                    prediction[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE]),

                "FPE_" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: tf.compat.v1.metrics.false_positives(
                    labels[IMAGE_CLEF_KEYS.CTR_CALCIFICATION],
                    prediction[IMAGE_CLEF_KEYS.CTR_CALCIFICATION]),

                "FPE_" + IMAGE_CLEF_KEYS.CTR_PLEURISY: tf.compat.v1.metrics.false_positives(
                    labels[IMAGE_CLEF_KEYS.CTR_PLEURISY],
                    prediction[IMAGE_CLEF_KEYS.CTR_PLEURISY]),

                "FPE_" + IMAGE_CLEF_KEYS.CTR_CAVERNS: tf.compat.v1.metrics.false_positives(
                    labels[IMAGE_CLEF_KEYS.CTR_CAVERNS],
                    prediction[IMAGE_CLEF_KEYS.CTR_CAVERNS]),

                "FNE_" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: tf.compat.v1.metrics.false_negatives(
                    labels[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED],
                    prediction[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED]),

                "FNE_" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: tf.compat.v1.metrics.false_negatives(
                    labels[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED],
                    prediction[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED]),

                "FNE_" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: tf.compat.v1.metrics.false_negatives(
                    labels[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE],
                    prediction[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE]),

                "FNE_" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: tf.compat.v1.metrics.false_negatives(
                    labels[IMAGE_CLEF_KEYS.CTR_CALCIFICATION],
                    prediction[IMAGE_CLEF_KEYS.CTR_CALCIFICATION]),

                "FNE_" + IMAGE_CLEF_KEYS.CTR_PLEURISY: tf.compat.v1.metrics.false_negatives(
                    labels[IMAGE_CLEF_KEYS.CTR_PLEURISY],
                    prediction[IMAGE_CLEF_KEYS.CTR_PLEURISY]),

                "FNE_" + IMAGE_CLEF_KEYS.CTR_CAVERNS: tf.compat.v1.metrics.false_negatives(
                    labels[IMAGE_CLEF_KEYS.CTR_CAVERNS],
                    prediction[IMAGE_CLEF_KEYS.CTR_CAVERNS]),

                ##metrics.mean just do the trick in order to show the eval loss per task
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.SEVERITY: tf.compat.v1.metrics.mean(av_per_task_loss[IMAGE_CLEF_KEYS.SEVERITY]),
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: tf.compat.v1.metrics.mean(
                    av_per_task_loss[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED]),
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: tf.compat.v1.metrics.mean(
                    av_per_task_loss[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED]),
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: tf.compat.v1.metrics.mean(
                    av_per_task_loss[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE]),
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: tf.compat.v1.metrics.mean(
                    av_per_task_loss[IMAGE_CLEF_KEYS.CTR_CALCIFICATION]),
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.CTR_PLEURISY: tf.compat.v1.metrics.mean(
                    av_per_task_loss[IMAGE_CLEF_KEYS.CTR_PLEURISY]),
                "LOSS_EVAL_" + IMAGE_CLEF_KEYS.CTR_CAVERNS: tf.compat.v1.metrics.mean(
                    av_per_task_loss[IMAGE_CLEF_KEYS.CTR_CAVERNS])}

        if mode == tf.estimator.ModeKeys.EVAL:
            eval_summary_hook = tf.estimator.SummarySaverHook(
                save_steps=500,
                output_dir=os.path.join(self.model_dir, "eval_validation"),
                summary_op=tf.compat.v1.summary.merge_all())

            if self.aim == Aim.CLASS_AND_SEG:
                eval_metric_ops_seg = {**eval_metric_ops_seg,
                                       **{"LOSS_SEG": tf.compat.v1.metrics.mean(av_per_task_loss['Loss_seg'])}}
                eval_metric_ops = {**eval_metric_ops_seg, **eval_metric_ops_clf}
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops,
                                                  evaluation_hooks=[eval_summary_hook])
            elif self.aim == Aim.CLASSIFICATION:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops_clf,
                                                  evaluation_hooks=[eval_summary_hook])
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops_seg,
                                                  evaluation_hooks=[eval_summary_hook])

        # Configure the Training Op (for TRAIN mode)
        multiple_opts = params['multiple_opts']
        optimizer_params_list = params['optimizer_params']
        learning_rate = params['learning_rate']
        optimizer = params['optimizer']

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        global_step = tf.compat.v1.train.get_global_step()
        with tf.control_dependencies(update_ops):
            if multiple_opts:
                train_cnn_op = get_subnet_estimator("cnn_features", loss, optimizer, learning_rate,
                                                    optimizer_params_list, step=global_step)
                train_multi_op = get_subnet_estimator("dense_shared_features", loss, optimizer, learning_rate,
                                                      optimizer_params_list, step=global_step)
                train_task_ops = [get_subnet_estimator(
                    "multi_task_features/" + task_feats + "_features", loss, optimizer, learning_rate,
                    optimizer_params_list)
                    for task_feats in IMAGE_CLEF_KEYS.keys_as_list()]
                train_op = tf.group([train_cnn_op, train_multi_op] + train_task_ops)
            else:
                if len(optimizer_params_list) == 0:
                    optimizer = optimizer(learning_rate=learning_rate)
                else:
                    new_list_params = [learning_rate] + optimizer_params_list
                    optimizer = optimizer(*new_list_params)

                # TODO check. This is the original before stuck at mirrorstrategy at multiple gpus
                # train_op = optimizer.minimize(loss, global_step=global_step)
                minimize_op = optimizer.minimize(loss=loss, global_step=tf.compat.v1.train.get_global_step())
                train_op = tf.group(minimize_op, update_ops)

            # train_summary_hook = tf.train.SummarySaverHook(
            #     save_steps=500,
            #     output_dir=os.path.join(self.model_dir, "eval_validation"),
            #     summary_op=tf.summary.merge_all())

            if self.aim == Aim.CLASS_AND_SEG:
                eval_metric_ops_seg = {**eval_metric_ops_seg,
                                       **{"LOSS_SEG": tf.compat.v1.metrics.mean(av_per_task_loss['Loss_seg'])}}
                eval_metric_ops = {**eval_metric_ops_seg, **eval_metric_ops_clf}
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
                                                  #training_hooks=[train_summary_hook])
            elif self.aim == Aim.CLASSIFICATION:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
