#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:20:00 2019

@author: Pedro M. Gordaliza
"""

import numpy as np

import tensorflow as tf
from utils import IMAGE_CLEF_KEYS


class TASK_TYPE:
    BINARY = 'Binary'
    REGRESSION = 'Regression'


def focal_loss_softmax(labels, logits, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `labels`
    """
    y_pred = tf.nn.softmax(logits, dim=-1)  # [batch_size,num_classes]
    labels = tf.one_hot(labels, depth=y_pred.shape[1])
    L = -labels * ((1 - y_pred) ** gamma) * tf.log(y_pred)
    L = tf.reduce_sum(L, axis=1)
    return L


def kendall_gal_loss_clf_sm_fl(labels, mask, clf_logits, seg_logits, log_vars_clf, log_var_seg, eps=1e-7):
    task_types = {IMAGE_CLEF_KEYS.SEVERITY: TASK_TYPE.REGRESSION,
                  IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CALCIFICATION: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_PLEURISY: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CAVERNS: TASK_TYPE.BINARY}
    loss, loss_acc = kendall_gal_loss_sm_fl(labels, clf_logits, task_types, log_vars=log_vars_clf)
    return tf.reduce_mean(loss), loss_acc


def kendall_gal_loss_sm_fl(ys_true, ys_pred, ys_tasks_type, log_vars=None, gamma=2):
    def specific_task_loss(y_true, y_pred, log_var_loss, name, task_type):
        y_true = tf.cast(y_true, tf.float32)
        precision = tf.exp(-log_var_loss, name='PRECISION')
        if task_type is TASK_TYPE.BINARY:
            y_true_hot = tf.one_hot(tf.cast(y_true[:, 0], tf.int32), depth=2, name='y_hot')
            y_pred = tf.nn.softmax(y_pred * precision, dim=-1)
            L = tf.reduce_sum(-y_true_hot * ((1 - y_pred) ** gamma) * tf.log(y_pred),
                              axis=1, keep_dims=True, name='TASK_LOSS_' + name)
            # L should be (bs,1)
            return L
        else:
            return tf.add(precision * tf.pow(y_true - y_pred, 2) / 2., log_var_loss / 2, name='TASK_LOSS_' + name)

    shape = tf.shape(ys_true[IMAGE_CLEF_KEYS.CTR_CAVERNS])
    loss = tf.zeros((shape[0], 1))
    loss_acc = {}
    for i, log_var_name in enumerate(log_vars):
        # log_var = log_vars[log_var_name]
        task_loss = specific_task_loss(ys_true[log_var_name], ys_pred[log_var_name], log_vars[log_var_name],
                                       log_var_name, task_type=ys_tasks_type[log_var_name])

        print('Task_loss' + log_var_name, task_loss)
        mean_per_task_loss = tf.reduce_mean(task_loss, name='TASK_LOSS_Mean' + log_var_name)
        tf.summary.scalar('loss_' + log_var_name, mean_per_task_loss)

        loss = tf.add(loss, task_loss, name='LOSS_ADD_' + log_var_name)
        # 'Loss should be (bs,1)', loss
        loss_acc[log_var_name] = mean_per_task_loss
    # 'Loss should be (bs,1) at the END', loss)
    return loss, loss_acc

    # return tf.reduce_mean(loss, name='REDUCED_LOSS'), loss_acc


def uncertainty_loss_sm_fl(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_seg_unc(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf_sm_fl(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc['Loss_seg'] = seg_loss
    return (seg_loss + clf_loss) / 2., loss_acc


def uncertainty_loss_sm_fl_2(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_seg(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf_sm_fl(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc['Loss_seg'] = seg_loss
    return (seg_loss + clf_loss) / 2., loss_acc


def uncertainty_loss_sm_fl_c(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_per_sample(labels, mask, clf_logits, seg_logits, log_vars_clf,
                                                          log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf_sm_fl(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc['Loss_seg'] = seg_loss
    print("LEN log_vars", len(log_vars_clf))
    n_tasks = len(log_vars_clf) + 1.0  # the 1.0 is the segmentation task. Floats to avoid conversions
    result = seg_loss + clf_loss  # / n_tasks
    print("RESULT", result)
    # mean over batch size
    return tf.reduce_mean(result), loss_acc


def uncertainty_loss_sm(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_seg_unc(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf_sm(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc['Loss_seg'] = seg_loss
    return (seg_loss + clf_loss) / 2., loss_acc


def uncertainty_loss(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_seg_unc(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc.append(seg_loss)
    return (seg_loss + clf_loss) / 2., loss_acc


def uncertainty_loss_2(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_seg_unc(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf2(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc.append(seg_loss)
    return (seg_loss + clf_loss) / 2., loss_acc


def uncertainty_loss_reg(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    seg_loss, seg_loss = sigmoid_cross_entropy_seg_unc(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    clf_loss, loss_acc = kendall_gal_loss_clf_reg(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg)
    loss_acc.append(seg_loss)
    return (seg_loss + clf_loss) / 2., loss_acc


def sigmoid_cross_entropy_seg_unc(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    loss = tf.losses.sigmoid_cross_entropy(mask, seg_logits, reduction=tf.losses.Reduction.NONE)

    precision = tf.exp(-log_vars_seg)
    task_loss = tf.multiply(precision, loss) + log_vars_seg / 2.
    tf.summary.histogram('uncertainty_seg', task_loss)

    loss = tf.reduce_mean(task_loss)
    return loss, loss


def sigmoid_cross_entropy_seg(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    # loss = tf.losses.sigmoid_cross_entropy(mask, seg_logits)
    loss = tf.losses.sigmoid_cross_entropy(mask, seg_logits)
    print('sigmoid cross entropy loss', loss)
    return loss, loss


def sigmoid_cross_entropy_per_sample(labels, mask, clf_logits, seg_logits, log_vars_clf, log_vars_seg):
    loss = tf.losses.sigmoid_cross_entropy(mask, seg_logits, reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(loss, axis=[1, 2, 3], name='CE_per_sample')
    print('sigmoid cross entropy loss', loss)
    return loss, loss


def kendall_gal_loss_clf2(labels, mask, clf_logits, seg_logits, log_vars_clf, log_var_seg, eps=1e-7):
    task_types = {IMAGE_CLEF_KEYS.SEVERITY: TASK_TYPE.REGRESSION,
                  IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CALCIFICATION: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_PLEURISY: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CAVERNS: TASK_TYPE.BINARY}
    labels = labels[:, :-1]

    loss = 0.
    loss_acc = []
    for i, log_var_name in enumerate(log_vars_clf):  # TODO REVISAR ####################################
        log_var = log_vars_clf[log_var_name]
        precision = tf.exp(-log_var, name='PRECISION')
        wo_loss = tf.abs(labels[log_var_name] - clf_logits[log_var_name]) if task_types[
                                                                                 log_var_name] == TASK_TYPE.REGRESSION else \
            tf.losses.sigmoid_cross_entropy(labels[log_var_name], clf_logits[log_var_name])
        spec_task_loss = (precision / 2.) * tf.abs(labels[log_var_name] - clf_logits[log_var_name]) + (log_var / 2.) \
            if task_types[i][0] == TASK_TYPE.REGRESSION else \
            tf.losses.sigmoid_cross_entropy(labels[log_var_name], clf_logits[log_var_name] * precision)
        tf.summary.scalar('wo/var_loss_' + log_var_name, tf.reduce_mean(wo_loss))

        task_loss = spec_task_loss

        mean_per_task_loss = tf.reduce_mean(task_loss, name='TASK_LOSS_' + log_var_name)
        tf.summary.scalar('loss_' + str(log_var_name), mean_per_task_loss)  # Show in tensorboard
        # mean_task_loss = tf.reduce_mean(task_loss) #Probatura
        loss = tf.add(loss, task_loss, name='LOSS_ADD_' + log_var_name)
        loss_acc.append(mean_per_task_loss)
    return tf.reduce_mean(loss, name='REDUCED_LOSS'), loss_acc


def kendall_gal_loss_clf_reg(labels, mask, clf_logits, seg_logits, log_vars_clf, log_var_seg, eps=1e-7):
    task_types = {IMAGE_CLEF_KEYS.SEVERITY: TASK_TYPE.REGRESSION,
                  IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CALCIFICATION: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_PLEURISY: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CAVERNS: TASK_TYPE.BINARY}
    return kendall_gal_loss(labels, clf_logits, task_types, log_vars=log_vars_clf, eps=eps)


def kendall_gal_loss_clf(labels, mask, clf_logits, seg_logits, log_vars_clf, log_var_seg, eps=1e-7):
    task_types = {IMAGE_CLEF_KEYS.SEVERITY: TASK_TYPE.REGRESSION,
                  IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CALCIFICATION: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_PLEURISY: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CAVERNS: TASK_TYPE.BINARY}
    return kendall_gal_loss(labels, clf_logits, task_types, log_vars=log_vars_clf, eps=eps)


def kendall_gal_loss_clf_sm(labels, mask, clf_logits, seg_logits, log_vars_clf, log_var_seg, eps=1e-7):
    task_types = {IMAGE_CLEF_KEYS.SEVERITY: TASK_TYPE.REGRESSION,
                  IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CALCIFICATION: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_PLEURISY: TASK_TYPE.BINARY,
                  IMAGE_CLEF_KEYS.CTR_CAVERNS: TASK_TYPE.BINARY}
    return kendall_gal_loss_sm(labels, clf_logits, task_types, log_vars=log_vars_clf, eps=eps)


def kendall_gal_loss_sm(ys_true, ys_pred, ys_tasks_type, log_vars=None, eps=1e-7):
    """
    Parameters
    -----------
    ys_task_types : Must be a list with a tuple in eac postion containing the variable type (var[0]) and the name (var[1])
        A distribution with shape: [batch_size, ....], (any dimensions).
        :type ys_true: object
        :param ys_true:
        :param log_vars:
    """

    def specific_task_loss(y_true, y_pred, task_type):
        y_true = tf.cast(y_true, tf.float32)
        if task_type is TASK_TYPE.BINARY:
            print('y_trye', y_true)
            print('y_trye', y_true[:, 0])
            print('y_pred', y_pred)
            y_true_hot = tf.one_hot(tf.cast(y_true[:, 0], tf.int32), depth=2, name='y_hot')

            print('y_true_hot', y_true_hot)
            return tf.losses.softmax_cross_entropy(y_true_hot, y_pred, reduction=tf.losses.Reduction.NONE)
        else:
            return tf.pow(y_true - y_pred, 2) / np.array(2, dtype=np.float32)

    shape = tf.shape(ys_true[IMAGE_CLEF_KEYS.CTR_CAVERNS])
    loss = tf.zeros((shape[0], 1))
    loss_acc = {}
    for i, log_var_name in enumerate(log_vars):
        log_var = log_vars[log_var_name]
        precision = tf.exp(-log_var, name='PRECISION')

        spec_task_loss = specific_task_loss(ys_true[log_var_name], ys_pred[log_var_name],
                                            task_type=ys_tasks_type[log_var_name])
        tf.summary.scalar('wo/var_loss_' + log_var_name, tf.reduce_mean(spec_task_loss))
        task_loss = tf.multiply(precision, spec_task_loss, name='TASK_LOSS_' + log_var_name) + log_var / 2.

        mean_per_task_loss = tf.reduce_mean(task_loss, name='TASK_LOSS_Mean' + log_var_name)
        tf.summary.scalar('loss_' + log_var_name, mean_per_task_loss)  # Show in tensorboard
        # mean_task_loss = tf.reduce_mean(task_loss) #Probatura
        loss = tf.add(loss, task_loss, name='LOSS_ADD_' + log_var_name)
        loss_acc[log_var_name] = mean_per_task_loss
    return tf.reduce_mean(loss, name='REDUCED_LOSS'), loss_acc


def kendall_gal_loss(ys_true, ys_pred, ys_tasks_type, log_vars=None, eps=1e-7):
    """
    Parameters
    -----------
    ys_task_types : Must be a list with a tuple in eac postion containing the variable type (var[0]) and the name (var[1])
        A distribution with shape: [batch_size, ....], (any dimensions).
        :type ys_true: object
        :param ys_true: 
        :param log_vars:
    """

    def specific_task_loss(y_true, y_pred, task_type):
        y_true = tf.cast(y_true, tf.float32)
        if task_type is TASK_TYPE.BINARY:
            return tf.losses.sigmoid_cross_entropy(y_true, y_pred, reduction=tf.losses.Reduction.NONE) + eps
        else:
            return tf.pow(y_true - y_pred, 2) / np.array(2, dtype=np.float32) + eps

    loss = 0.
    loss_acc = []
    for i, log_var_name in enumerate(log_vars):
        log_var = log_vars[log_var_name]
        precision = tf.exp(-log_var, name='PRECISION')

        spec_task_loss = specific_task_loss(ys_true[log_var_name], ys_pred[log_var_name],
                                            task_type=ys_tasks_type[log_var_name])
        tf.summary.scalar('wo/var_loss_' + log_var_name, tf.reduce_mean(spec_task_loss))
        task_loss = tf.multiply(precision, spec_task_loss, name='TASK_LOSS_' + log_var_name) + log_var / 2.

        mean_per_task_loss = tf.reduce_mean(task_loss, name='TASK_LOSS_' + log_var_name)
        tf.summary.scalar('loss_' + log_var_name, mean_per_task_loss)  # Show in tensorboard
        # mean_task_loss = tf.reduce_mean(task_loss) #Probatura
        loss = tf.add(loss, task_loss, name='LOSS_ADD_' + ys_tasks_type[i][1])
        loss_acc.append(mean_per_task_loss)
    return tf.reduce_mean(loss, name='REDUCED_LOSS'), loss_acc
