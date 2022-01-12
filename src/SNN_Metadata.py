#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 15:21:00 2019

@author: Pedro M. Gordaliza
"""
import os
import argparse
import numpy as np
import pandas as pd
import glob

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from utils import IMAGE_CLEF_KEYS, IMAGE_CLEF_META
from selu_utils import dropout_selu

#Hyperparameters considered for self-normalizing networks
hidden_units = [1024, 512, 256]
hidden_layers = [3, 4, 8, 16, 32]
learning_rates = [0.01, 0.1]
dropout_rates = [0.05, 0.1]


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)



def get_anotations_labels_decode(img_path, pandas_report, from_tensor=True):
    with file_io.FileIO(pandas_report, mode='r') as f:
        pandas_report = pd.read_csv(f)
    annotatations_pandas = pandas_report

    file_reference = img_path.decode("utf-8") if from_tensor else img_path
    img_fields = file_reference.split(os.sep)[-1]
    name = img_fields + '.gz'

    anotations = annotatations_pandas[annotatations_pandas['Filename'] == name]
    #annotation_severity = anotations[IMAGE_CLEF_KEYS.SEVERITY].values[0]
    annotation_left_lung = anotations[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED].values[0]
    annotation_right_lung = anotations[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED].values[0]
    annotation_lung_capacity = anotations[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE].values[0]
    anotation_calcification = anotations[IMAGE_CLEF_KEYS.CTR_CALCIFICATION].values[0]
    anotation_pleurisy = anotations[IMAGE_CLEF_KEYS.CTR_PLEURISY].values[0]
    anotation_caverns = anotations[IMAGE_CLEF_KEYS.CTR_CAVERNS].values[0]
    anotation_svr_severity = anotations[IMAGE_CLEF_KEYS.SVR_SEVERITY].values[0]
    anotation_svr_severity = 1 if anotation_svr_severity == 'LOW' else 0  # "LOW" (scores 4 and 5) and "HIGH" (scores 1, 2 and 3).

    y = np.array([annotation_left_lung,
                  annotation_right_lung,
                  annotation_lung_capacity,
                  anotation_calcification,
                  anotation_pleurisy,
                  anotation_caverns,
                  anotation_svr_severity])

    meta_disabilty = anotations[IMAGE_CLEF_META.DISABILITY].values[0]
    meta_relapse =  anotations[IMAGE_CLEF_META.RELAPSE].values[0]
    meta_tb_symps =  anotations[IMAGE_CLEF_META.TB_SYMPTONS].values[0]
    meta_comorbidity =  anotations[IMAGE_CLEF_META.COMORBIDITY].values[0]
    meta_bacilary =  anotations[IMAGE_CLEF_META.BACILARY].values[0]
    meta_drug_resistant = anotations[IMAGE_CLEF_META.DRUG_RESISTANT].values[0]
    meta_higher_education = anotations[IMAGE_CLEF_META.HIGHER_EDUCATION].values[0]
    meta_ex_prisoner = anotations[IMAGE_CLEF_META.EX_PRISONER].values[0]
    meta_alcoholic = anotations[IMAGE_CLEF_META.ALCOHOLIC].values[0]
    meta_smoker = anotations[IMAGE_CLEF_META.SMOKER].values[0]

    meta_information = np.array([meta_disabilty,
                                 meta_relapse,
                                 meta_tb_symps,
                                 meta_comorbidity,
                                 meta_bacilary,
                                 meta_drug_resistant,
                                 meta_higher_education,
                                 meta_ex_prisoner,
                                 meta_alcoholic,
                                 meta_smoker])

    name_r = str(os.path.basename(img_path))

    return meta_information.astype(np.float32), y.astype(np.float32), name_r

def set_shapes_and_dict(metadata, y, name, shapes=[[10], [7], []]):
    print('set shapes', y)
    metadata.set_shape(shapes[0])
    y.set_shape(shapes[1])
    name.set_shape(shapes[2])
    return {'x': metadata}, {'labels': {'y': y, 'Name': name}}


def input_fn(file_references, mode,
             reports_path="/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_metaData_extra.csv"):

    dataset = tf.data.Dataset.list_files(file_references)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(50)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(get_anotations_labels_decode, [filename, reports_path],
                                                                [tf.float32, tf.float32, tf.string])), num_parallel_calls=6)
    dataset = dataset.map(set_shapes_and_dict, num_parallel_calls=6)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(None)
    dataset = dataset.prefetch(1)

    iterator =dataset.make_one_shot_iterator()

    next_dict = iterator.get_next()

    #Set runhook to initialize iterator
    iterator_initializer_hook = IteratorInitializerHook()
    iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)

    #Return batched (features, labels)
    print('DICT', next_dict)
    #print( 'next_dict[labels]',  next_dict['labels'])
    return next_dict#, next_dict['labels']

def model_fn(features, labels, mode, params):
    def specific_block(feats):
        specific_units = [128, 64, 32, 16]
        for n_units in specific_units:
            feats = tf.layers.dense(feats, n_units, tf.nn.selu)
            feats = dropout_selu(feats, dropout_rate)
        return feats

    print('FEATURES', features)
    print('LABELS', labels)
    x = features['x']
    y = labels['labels']['y']
    name = labels['labels']['Name']
    hidden_layers = params['hidden_layers']
    hidden_units = params['hidden_units']
    learning_rate = params['learning_rate']
    dropout_rate = params['dropout_rate']

    for l in range(hidden_layers):
        x = tf.layers.dense(x, hidden_units, activation=tf.nn.selu)
        x = dropout_selu(x, dropout_rate)

    logits = [tf.layers.dense(specific_block(x), units=1) for task in range(7)]

    #logits = tf.layers.dense(x, 7)
    logits = tf.concat(logits, axis=1)

    probabilities = tf.nn.sigmoid(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        pass

    loss = tf.losses.sigmoid_cross_entropy(y, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.train.get_global_step()
        with tf.control_dependencies(update_ops):
            with tf.variable_scope("train_ops"):
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    print('LABELS', y)
    print('PROBS', probabilities)

    AUC_CTR_left_lung_affected = tf.metrics.auc(y[:, 0], probabilities[:, 0])
    AUC_CTR_right_lung_affected = tf.metrics.auc(y[:, 1], probabilities[:, 1])
    AUC_CTR_lung_capacity_decrease = tf.metrics.auc(y[:, 2], probabilities[:, 2])
    AUC_CTR_calcification = tf.metrics.auc(y[:, 3], probabilities[:, 3])
    AUC_CTR_pleurisy = tf.metrics.auc(y[:, 4], probabilities[:, 4])
    AUC_CTR_caverns = tf.metrics.auc(y[:, 5], probabilities[:, 5])
    AUC_SVR_severity = tf.metrics.auc(y[:, 6], probabilities[:, 6])

    eval_metrics = {"AUC_" + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: AUC_CTR_left_lung_affected,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: AUC_CTR_right_lung_affected,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: AUC_CTR_lung_capacity_decrease,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: AUC_CTR_calcification,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_PLEURISY: AUC_CTR_pleurisy,
                "AUC_" + IMAGE_CLEF_KEYS.CTR_CAVERNS: AUC_CTR_caverns,
                "AUC_" + IMAGE_CLEF_KEYS.SVR_SEVERITY: AUC_SVR_severity}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)


def main(argv):
    np.random.seed(42)
    tf.set_random_seed(42)
    base_dir = args.data_dir
    train_dir = args.train_imgs_dir
    val_imgs_dir = args.val_imgs_dir
    reports_path = args.reports_path


    for hidden_layer in hidden_layers:
        for hidden_unit in hidden_units:
            for learning_rate in learning_rates:
                for dropout_rate in dropout_rates:
                    params = {'hidden_layers': hidden_layer, 'hidden_units': hidden_unit, 'learning_rate':learning_rate,
                              'dropout_rate': dropout_rate}



                    model_dir = os.path.join(base_dir, 'FNN-SNN-EXPERIMENTS_2', 'HL_'+str(hidden_layer)+'_HU_'
                                             + str(hidden_unit)+'_LR_'+str(learning_rate)+'_DR_'+str(dropout_rate))

                    my_checkpointing_config = tf.estimator.RunConfig(
                        keep_checkpoint_max=5,  # Retain the 5 most recent checkpoints.
                        save_checkpoints_steps=100,
                        save_summary_steps=20
                    )

                    classifier = tf.estimator.Estimator(model_fn=model_fn,
                                                        model_dir=model_dir,
                                                        config=my_checkpointing_config,
                                                        params=params)

                    def train_input_fn():
                        return input_fn(glob.glob(train_dir + '*' + 'nii'), tf.estimator.ModeKeys.TRAIN, reports_path=reports_path)

                    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=10000)

                    def eval_input_fn():
                        return input_fn(glob.glob(val_imgs_dir + '*' + 'nii'), tf.estimator.ModeKeys.EVAL, reports_path=reports_path)

                    eval_spec = tf.estimator.EvalSpec(eval_input_fn, name='validation', throttle_secs=500)
                    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='GCS or local path to training data',
        required=True
    )

    parser.add_argument(
        '--train_imgs_dir',
        help='GCS or local path to training data',
        required=True
    )


    parser.add_argument(
        '--val_imgs_dir',
        help='GCS or local path to training data',
        required=True
    )

    parser.add_argument(
        '--reports_path',
        help='GCS or local path to training data',
        type=str,
        default=None
    )


    args = parser.parse_args()
    arguments = args.__dict__
    tf.app.run()
