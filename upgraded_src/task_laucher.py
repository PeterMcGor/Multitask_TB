#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:20:00 2019

@author: Pedro M. Gordaliza
"""
# TODO INCLUDE XLA FLAG!!!!!!!!!!!
import os
import configargparse
# import argparse
import glob
import numpy as np
import time
import sys
import pandas as pd

import tensorflow as tf

from abstract_reader import Reader, my_service
# from EvalResultsExporter import EvalResultsExporter

from model import Model, Aim, INIT_BIAS
from losses import kendall_gal_loss, kendall_gal_loss_clf, sigmoid_cross_entropy_seg_unc, sigmoid_cross_entropy_seg, \
    uncertainty_loss, kendall_gal_loss_clf2, uncertainty_loss_reg, kendall_gal_loss_clf_sm, uncertainty_loss_sm, \
    uncertainty_loss_sm_fl, uncertainty_loss_sm_fl_2, uncertainty_loss_sm_fl_c, kendall_gal_loss_clf_sm_fl
from utils import read_fn, read_fn_lbels, get_image_dimension, IMAGE_CLEF_KEYS, get_itk_image, adamW
from shared_blocks import cnn_shared_features, fully_connected_block, specific_cnn_by_manifestation, vnet_decoder_style

AIM_DICT = {Aim.CLASSIFICATION: Aim.CLASSIFICATION,
            Aim.SEGMENTATION: Aim.SEGMENTATION,
            Aim.CLASS_AND_SEG: Aim.CLASS_AND_SEG}

LOSS_FN_DICT = {'RMSE': tf.compat.v1.losses.mean_squared_error,
                'SIGMOID_CROSS_ENTROPY': sigmoid_cross_entropy_seg,
                'KENDALL_GAL_HOMO': kendall_gal_loss,
                'KENDALL_GAL_CLF': kendall_gal_loss_clf,
                'SIGMOID_CROSS_ENTROPY_UNC': sigmoid_cross_entropy_seg_unc,
                'KENDALL_GAL_CLF_SM_FL': kendall_gal_loss_clf_sm_fl,
                'UNCERTAINTY_LOSS': uncertainty_loss,
                'ALT_CLF': kendall_gal_loss_clf2,
                'UNCERTAINTY_LOSS_REG': uncertainty_loss_reg,
                'KENDALL_GAL_CL_SM': kendall_gal_loss_clf_sm,
                'UNCERTAINTY_LOSS_SM': uncertainty_loss_sm,
                'UNCERTAINTY_LOSS_SM_FL': uncertainty_loss_sm_fl,
                'UNCERTAINTY_LOSS_SM_FL_2': uncertainty_loss_sm_fl_2,
                'UNCERTAINTY_LOSS_SM_FL_C': uncertainty_loss_sm_fl_c}

ACTIVATION_FN_DICT = {'relu': tf.nn.relu,
                      'relu6': tf.nn.relu6,
                      'prelu': tf.nn.leaky_relu,
                      'selu': tf.nn.selu}

SHARED_BLOCK_DICT = {'cnn_shared': cnn_shared_features,
                     'fnn_shared': fully_connected_block,
                     'specific_cnn': specific_cnn_by_manifestation,
                     'vnet_decoder': vnet_decoder_style}

OPTIMIZER_DICT = {'SGD': tf.compat.v1.train.GradientDescentOptimizer,
                  'ADAM': tf.compat.v1.train.AdamOptimizer,
                  'ADAMW': adamW,
                  'RMSPROP': tf.compat.v1.train.RMSPropOptimizer,
                  'MOMENTUM': tf.compat.v1.train.MomentumOptimizer}

MODEL_DICT = {'UNet': Model.UNet,
              'VNet': Model.VNet}

# os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' #TODO problems with mirror strategy
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.compat.v1.random.set_random_seed(seed=42)




def get_distribution_strategy(num_gpus, all_reduce_alg=None):
    """Return a DistributionStrategy for running the model.
    Args:
      num_gpus: Number of GPUs to run this model.
      all_reduce_alg: Specify which algorithm to use when performing all-reduce.
        See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
        If None, DistributionStrategy will choose based on device topology.
    Returns:
      tf.contrib.distribute.DistibutionStrategy object.
    """
    if num_gpus == 0:
        return tf.distribute.OneDeviceStrategy("device:CPU:0")
    elif num_gpus == 1:
        return tf.distribute.OneDeviceStrategy("device:GPU:0")
    else:
        if all_reduce_alg:
            return tf.contrib.distribute.MirroredStrategy(
                num_gpus=num_gpus,
                cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
                    all_reduce_alg, num_packs=num_gpus))
        else:
            return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def main(argv):
    np.random.seed(42)
    tf.compat.v1.set_random_seed(42)
    print('ARGS', args)

    serialized = False
    classification_task = args.classification_task
    loss_fn = LOSS_FN_DICT[args.loss_fn]

    # data_dir = args.data_dir
    train_dir = args.train_imgs_dir
    val_dir = args.val_imgs_dir
    train_masks_dir = args.train_masks_dir
    val_masks_dir = args.val_masks_dir
    reports_path = args.reports_path
    num_parallel_calls = args.num_parallel_calls
    t_tag = args.from_tag if args.from_tag != 'No' else time.strftime("%Y%m%d%H%M%S")

    architecture = MODEL_DICT[args.architecture]
    activation_fn = ACTIVATION_FN_DICT[args.activation]
    activation_dense_fn = ACTIVATION_FN_DICT[args.activation_dense]
    keep_prob = args.keep_prob
    num_channels = args.num_channels
    num_levels = args.num_levels
    learning_rate = args.learning_rate
    optimizer = OPTIMIZER_DICT[args.optimizer]
    print("PARAMS", args.optimizer_params)
    optimizer_params = [float(item) for item in args.optimizer_params[0].split(",")]
    print('OPT PARAMS', optimizer_params)
    augment_train = args.augment_train
    # augment_validation = args.augment_validation
    aleatoric_uncertainty = args.aleatoric_uncertainty
    multiple_opts = args.multiple_opts
    num_gpus = args.num_gpus
    normalize_input = args.normalize_input
    hierarchy = args.hierarchy
    shared_fn = SHARED_BLOCK_DICT[args.shared_fn]
    augment_dificult = args.augment_dificult
    employ_meta_info = args.employ_meta_info
    steps_train = args.train_steps
    init_bias = args.init_bias
    bn_dense_block = args.bn_dense_block
    renorm_dense_block = args.renorm_dense_block
    arq_batch_normalization = args.architecture_batch_normalization
    arq_renormalization = args.architecture_renormalization
    activation_after_add = args.activation_after_add

    sep = '_'
    experiment = args.experiment
    model_dir = args.output_dir
    train_batch_size = args.train_batch_size
    optimizers_params_string = ''
    for p in optimizer_params:
        optimizers_params_string += '_' + str(p)

    model_dir = os.path.join(model_dir, experiment)

    os.makedirs(model_dir, exist_ok=True)
    parser.write_config_file(args, [os.path.join(model_dir, 'config_file_'+experiment+'.ini')])

    # if not os.path.exists(os.path.join(model_dir, 'params.log')):
    #     with open(os.path.join(model_dir, 'params.log'), 'x') as filehandle:
    #         filehandle.write('\n'.join(sys.argv[1:]))
    print('The fold model: ', experiment, 'would be stored at ', model_dir)

    # Config hardware strategy. Mirrors and estimators are the easiest wat for portable code between GCP and
    # local-machines
    mirrored_strategy = get_distribution_strategy(num_gpus)
    my_checkpointing_config = tf.estimator.RunConfig(
        # session_config=_get_session_config_from_env_var(),
        # save_checkpoints_secs = 30*60,  # Save checkpoints every 30 minutes.
        keep_checkpoint_max=5,  # Retain the 5 most recent checkpoints.
        save_checkpoints_steps=1000,
        save_summary_steps=100,
        train_distribute=mirrored_strategy,
        # eval_distribute=mirrored_strategy
        # session_config = config
    )

    # Just to take into account in GCP execution.
    # f_header = 'gs://the_shit/serialized_cropped_monkeys/'
    f_extension = ".tfrecords" if serialized else ".nii"
    print('Train_images_path', train_dir + '*' + f_extension)
    train_images_paths = glob.glob(train_dir + '*' + f_extension)
    train_buffer_size = args.train_buffer_size if args.train_buffer_size is not None else len(train_images_paths)
    val_images_paths = glob.glob(val_dir + '*' + f_extension)

    # Set volumes sizes #
    input_image_size = list(get_image_dimension(train_images_paths[0])) + [1]
    output_size = [8] if classification_task else input_image_size

    # #############################Set Classifier Model############################################
    ######################################################################################################
    model_params = {'model_dir': model_dir}
    # Set specific architecture params
    arq_params = {'architecture': architecture,
                  'num_channels': args.num_channels,
                  'num_levels': num_levels,
                  'keep_prob': keep_prob,
                  'activation': activation_fn,
                  'aleatoric_uncertainty': aleatoric_uncertainty,
                  'output_size': output_size,
                  'activation_dense_fn': activation_dense_fn,
                  'batch_normalization': arq_batch_normalization,
                  'renormalization': arq_renormalization,
                  'activation_after_add': activation_after_add
                  }

    shared_params = {'architecture': shared_fn,
                     'activation':  activation_dense_fn,
                     'input_size': np.array(input_image_size)/2**num_levels}

    params = {'activation_dense_fn': activation_dense_fn, 'aleatoric_uncertainty': aleatoric_uncertainty,
              'loss_fn': loss_fn, 'optimizer': optimizer, 'optimizer_params': optimizer_params,
              'learning_rate': learning_rate, 'multiple_opts': multiple_opts, 'hierarchy': hierarchy,
              'employ_meta_info': employ_meta_info, 'BN_dense_block': bn_dense_block,
              'renorm_dense_block': renorm_dense_block, 'init_bias': init_bias}

    model_clf = Model(classification_task, arq_params, shared_params, model_params)
    if shared_fn is SHARED_BLOCK_DICT['vnet_decoder']:
        model_clf.shared_block_params = {'activation':  activation_dense_fn,
                                         'v_net_model': model_clf.model}

    # The classifier is instantiated employing estimator, which are really helpful at cross configurations
    classifier = tf.estimator.Estimator(model_fn=model_clf.model_fn,
                                        model_dir=model_dir,
                                        config=my_checkpointing_config,
                                        params=params)

    # Set the input elements shape/dtype for readers
    reader_fn = read_fn_lbels if classification_task else read_fn
    shape_labels_read = output_size if classification_task else input_image_size

    # Set up a data reader to handle the file i/o for training data. Due to the way tf handle None type the easiest
    # workaround is to define Nones as strings. That why mask or y are defined as string when is needed
    if classification_task == Aim.CLASSIFICATION:
        reader_example_shapes = {'features': {'x': input_image_size, 'meta_info': [10]},
                                 'labels': {'Mask': input_image_size, 'y': shape_labels_read, 'Name': []}}
        reader_examples_types = {'features': {'x': tf.float32, 'meta_info': tf.float32},
                                 'labels': {'Mask': tf.string, 'y': tf.float32, 'Name': tf.string}}
    elif classification_task == Aim.SEGMENTATION:
        reader_example_shapes = {'features': {'x': input_image_size, 'meta_info': [10]},
                                 'labels': {'Mask': input_image_size, 'y': shape_labels_read, 'Name': []}}
        reader_examples_types = {'features': {'x': tf.float32, 'meta_info': tf.string},
                                 'labels': {'Mask': tf.float32, 'y': tf.string, 'Name': tf.string}}
    else:
        reader_example_shapes = {'features': {'x': input_image_size, 'meta_info': [10]},
                                 'labels': {'Mask': input_image_size, 'y': shape_labels_read, 'Name': []}}
        reader_examples_types = {'features': {'x': tf.float32, 'meta_info': tf.float32},
                                 'labels': {'Mask': tf.float32, 'y': tf.float32, 'Name': tf.string}}

    # Set the function (Reader) to feed with the data the classifier at each different MODE #
    reader = Reader(reader_fn, reader_examples_types)
    # input_fn_reader, qinit_hook
    input_fn_reader = reader.get_inputs(file_references=train_images_paths,
                                        mode=tf.estimator.ModeKeys.TRAIN,
                                        example_shapes=reader_example_shapes,
                                        shuffle_cache_size=train_buffer_size,
                                        batch_size=train_batch_size,
                                        params={'extract_examples': False,
                                                'masks_dir': train_masks_dir,
                                                'reports_path': reports_path,
                                                'num_parallel_calls': num_parallel_calls,
                                                'augment_train': augment_train,
                                                'normalize_input': normalize_input,
                                                'augment_dificult': augment_dificult})
    # Set up hook to obtain real-time information
    tensors_to_log = {'Check_rand': 'Check_rand', 'MODE_NAME': 'MODE_NAME'}  #  'MEAN_OUT_LAYER':'MEAN_OUT_LAYER',  'MIN_OUT_LAYER': 'MIN_OUT_LAYER', 'MAX_OUT_LAYER':'MAX_OUT_LAYER''CE_per_sample': 'CE_per_sample'}
    levels = [3, 4]
    covs = [1, 2, 3]
    KPIS = ['MEAN_BN', 'MIN_BN', 'MAX_BN']
    KPIS_2 = ['MEAN_OUT_LAYER', 'MIN_OUT_LAYER', 'MAX_OUT_LAYER']
    # tensors_to_log = {'Check_rand': 'Check_rand', 'binary_logits':'binary_logits',
    #                   'concat_logits':'concat_logits', 'y_hot':'y_hot'}

    encoder_hooks = [{'cnn_features/vnet/encoder/level_'+str(l)+'/conv_'+str(c)+'/'+KPI:'cnn_features/vnet/encoder/level_'+str(l)+'/conv_'+str(c)+'/'+KPI} for l in levels for c in covs for KPI in KPIS]
    encoder_hooks_dct = {}
    for dct in encoder_hooks:
        encoder_hooks_dct = {**encoder_hooks_dct, **dct}
    dense_hooks = [{'multi_task_features/' + manifestation+'_features/'+KPI2: 'multi_task_features/' + manifestation+'_features/'+KPI2 for manifestation in IMAGE_CLEF_KEYS.keys_as_list() for KPI2 in KPIS_2}]
    dense_hooks_dct = {}
    for dct in dense_hooks:
        dense_hooks_dct = {**dense_hooks_dct, **dct}

    softmax_hooks = [{hook+manifestation: hook+manifestation
                      for manifestation in IMAGE_CLEF_KEYS.keys_as_list(just_binary=True)}
                     for hook in ['Softmax_', 'argmax_', 'Max_', 'out_prob_']]
    loss_hook = [{hook+manifestation: hook+manifestation for manifestation in IMAGE_CLEF_KEYS.keys_as_list()}
                 for hook in ['TASK_LOSS_', 'TASK_LOSS_Mean', 'LOSS_ADD_']]
    # x_hook = {'multi_task_features/'+manifestation+'_features/The_X_at_md_'+
    # manifestation:'multi_task_features/'+manifestation+'_features/The_X_at_'+
    # manifestation for manifestation in IMAGE_CLEF_KEYS.keys_as_list()}
    softmax_hooks_dct = {}
    loss_hooks_dct = {}
    for dct in softmax_hooks:
        softmax_hooks_dct = {**softmax_hooks_dct, **dct}
    for dct in loss_hook:
        loss_hooks_dct = {**loss_hooks_dct, **dct}

    tensors_to_log = {**tensors_to_log, **encoder_hooks_dct, **dense_hooks_dct}
    logging_hook = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
    train_spec = tf.estimator.TrainSpec(input_fn_reader, max_steps=steps_train, hooks=[logging_hook])

    # ### Set evaluation elements#####################
    reader_eval = Reader(reader_fn, reader_examples_types)
    eval_batch_size = min(len(val_images_paths), train_batch_size)

    # Reader (input_fn) to feed the estimator at evaluation time.
    # Similar to training except for the randomization and the augmenting

    input_fn_reader_val = reader_eval.get_inputs(file_references=val_images_paths,
                                                 mode=tf.estimator.ModeKeys.EVAL,
                                                 example_shapes=reader_example_shapes,
                                                 batch_size=eval_batch_size,
                                                 shuffle_cache_size=1,
                                                 params={'extract_examples': False,
                                                         'masks_dir': val_masks_dir,
                                                         'reports_path': reports_path,
                                                         'num_parallel_calls': num_parallel_calls,
                                                         'augment_train': augment_train,
                                                         'normalize_input': normalize_input})

    exporter = tf.estimator.LatestExporter('exporter', reader.serving_input_receiver_fn(reader_example_shapes),
                                           exports_to_keep=5)

    # EvalResultsExporter('eval_results')
    logging_hook_2 = tf.estimator.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=1)
    eval_spec = tf.estimator.EvalSpec(input_fn_reader_val, name='validation',
                                      start_delay_secs=10, throttle_secs=10,
                                      steps=np.ceil(len(val_images_paths) / eval_batch_size),
                                      exporters=[exporter],
                                      hooks=[logging_hook_2])

    # Dale candela#
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    # Start predictions
    predictor_fn = tf.contrib.predictor.from_estimator(classifier,
                                                       reader.serving_input_receiver_fn(reader_example_shapes))
    print('PREDICTOR_FN', predictor_fn)
    predictions_list = []
    for img, meta_info, name in my_service(val_images_paths, reports_path, normalize_input):
        print("Prediction for", name, "...")
        img = img[np.newaxis, :, :, :, :]
        meta_info = meta_info[np.newaxis, :]
        pred = predictor_fn({'x': img, 'meta_info': meta_info})

        if not os.path.exists(os.path.join(model_dir, 'results')):
            os.mkdir(os.path.join(model_dir, 'results'))

        if classification_task != Aim.SEGMENTATION:
            print('Making predictions...', pred)
            severity = pred[IMAGE_CLEF_KEYS.SEVERITY]
            left_lung_affected = pred[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED]
            right_lung_affected = pred[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED]
            lung_cap_decrease = pred[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE]
            calcification = pred[IMAGE_CLEF_KEYS.CTR_CALCIFICATION]
            pleurisy = pred[IMAGE_CLEF_KEYS.CTR_PLEURISY]
            caverns = pred[IMAGE_CLEF_KEYS.CTR_CAVERNS]
            binary_severity = pred[IMAGE_CLEF_KEYS.SVR_SEVERITY]
            predictions_list.append({'ID': name,
                                     IMAGE_CLEF_KEYS.SEVERITY: severity[0],
                                     IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: left_lung_affected[0],
                                     IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: right_lung_affected[0],
                                     IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: lung_cap_decrease[0],
                                     IMAGE_CLEF_KEYS.CTR_CALCIFICATION: calcification[0],
                                     IMAGE_CLEF_KEYS.CTR_PLEURISY: pleurisy[0],
                                     IMAGE_CLEF_KEYS.CTR_CAVERNS: caverns[0],
                                     IMAGE_CLEF_KEYS.SVR_SEVERITY: binary_severity[0]})

            pd.DataFrame(predictions_list).to_csv(os.path.join(model_dir, 'results', 'predictions.csv'))

        if classification_task != Aim.CLASSIFICATION:
            classes = pred['classes']
            preds = pred['probabilities']
            get_itk_image(img, os.path.join(model_dir, 'results', name+'_img.nii'))
            get_itk_image(preds, os.path.join(model_dir, 'results', name + '_preds.nii'))
            get_itk_image(classes.astype(np.uint8), os.path.join(model_dir, 'results', name+'_classes.nii'))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')
        # raise argparse.ArgumentTypeError('Boolean value expected.')


def none_or_str(value):
    if value in ('None', 'none'):
        return None
    return value


if __name__ == '__main__':
    parser = configargparse.ArgParser()

    parser.add('-c', '--my_config', required=True, is_config_file=True, help='config file path')
    parser.add('-e', '--experiment', required=True, help='Experiment name')

    parser.add_argument(
        '--train_imgs_dir',
        help='GCS or local path to training data',
        required=True
    )

    parser.add_argument(
        '--train_masks_dir',
        help='GCS or local path to training data',
        required=False,
        type=none_or_str,
        default=None
    )

    parser.add_argument(
        '--val_imgs_dir',
        help='GCS or local path to training data',
        required=True
    )

    parser.add_argument(
        '--val_masks_dir',
        help='GCS or local path to training data',
        required=False,
        type=none_or_str,
        default=None
    )

    parser.add_argument(
        '--reports_path',
        help='GCS or local path to training data',
        type=str,
        default=None
    )

    parser.add_argument(
        '--train_batch_size',
        help='Batch size for training steps',
        type=int,
        default=5
    )
    parser.add_argument(
        '--eval_batch_size',
        help='Batch size for evaluation steps',
        type=int,
        default=2
    )
    parser.add_argument(
        '--train_steps',
        help='Steps to run the training job for.',
        type=int,
        default=5
    )
    parser.add_argument(
        '--eval_steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=2,
        type=int
    )

    parser.add_argument(
        "--num_channels",
        help="Num Channels at the model [1,2,4,8,16]",
        default=16,
        choices=[1, 2, 4, 8, 16],
        type=int
    )

    parser.add_argument(
        "--num_levels",
        help="Num Channels at the model [1,2,4,8,16]",
        default=4,
        choices=[1, 2, 4, 8],
        type=int
    )

    parser.add_argument(
        "--loss_fn",
        help="Loss function",
        default='KENDALL_GAL_HOMO',
        choices=[k for k in LOSS_FN_DICT.keys()],
        type=str
    )

    parser.add_argument(
        "--shared_fn",
        help="shared block",
        default='cnn_shared',
        choices=[k for k in SHARED_BLOCK_DICT.keys()],
        type=str
    )

    parser.add_argument(
        "--keep_prob",
        help="Drop-out probability",
        default=0.5,
        type=float)

    parser.add_argument(
        "--num_parallel_calls",
        help="Parallel call at Dataset map",
        default=1,
        type=int
    )


    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        # GCS location to write checkpoints and export models',
        required=True
    )

    parser.add_argument(
        '--architecture',
        help='GCS location to write checkpoints and export models',
        default=Model.VNet,
        choices=[k for k in MODEL_DICT.keys()],
        required=True
    )
    parser.add_argument(
        '--architecture_batch_normalization',
        help='GCS location to write checkpoints and export models',
        default=False,
        required=False,
        type=str2bool
    )
    parser.add_argument(
        '--activation_after_add',
        help='GCS location to write checkpoints and export models',
        default=False,
        required=False,
        type=str2bool
    )
    parser.add_argument(
        '--architecture_renormalization',
        help='GCS location to write checkpoints and export models',
        default=False,
        required=False,
        type=str2bool
    )

    parser.add_argument(
        "--activation",
        default='relu6',
        choices=[k for k in ACTIVATION_FN_DICT.keys()],
        type=str
    )

    parser.add_argument(
        "--activation_dense",
        default='relu6',
        choices=[k for k in ACTIVATION_FN_DICT.keys()],
        type=str
    )

    parser.add_argument(
        "--from_tag",
        default=None,
        type=str
    )

    parser.add_argument(
        "--optimizer",
        default='ADAM',
        choices=[k for k in OPTIMIZER_DICT.keys()],
        type=str
    )

    parser.add_argument(
        '--optimizer_params',
        nargs='*',
        help='<Required> Set flag',
        default="",
        required=False)

    parser.add_argument(
        "--learning_rate",
        default=1e-5,
        type=float)

    parser.add_argument(
        "--aleatoric_uncertainty",
        default='Homoscedastic',
        choices=['Homoscedastic', 'Heterocedastic', 'Epistemic'],
        type=str
    )

    parser.add_argument(
        '--classification_task',
        help='',
        choices=[k for k in AIM_DICT.keys()],
        default=Aim.CLASSIFICATION,
        required=True,
        type=str
    )

    parser.add_argument(
        '--augment_train',
        help='augment_train_images',
        default=True,
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--normalize_input',
        help='whether normalise the input images between -1 and 1',
        default=True,
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--multiple_opts',
        help='',
        default=False,
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--hierarchy',
        help='',
        default=False,
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--employ_meta_info',
        help='',
        default='True',
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--bn_dense_block',
        help='',
        default='False',
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--renorm_dense_block',
        help='',
        default='False',
        required=False,
        type=str2bool
    )

    parser.add_argument(
        '--init_bias',
        help="Introduce bias at the beginning of the inference to avoid the model get stuck at average predictions. "
             "Check Focal Loss paper: https://arxiv.org/abs/1708.02002",
        default=INIT_BIAS.MINORITY,
        choices=[INIT_BIAS.MINORITY, INIT_BIAS.MAJORITY],
        type=str
    )

    parser.add_argument(
        '--augment_dificult',
        help='',
        default=1,
        required=False,
        type=int
    )

    parser.add_argument(
        '--train_buffer_size',
        help='How long to wait before running first evaluation',
        required=False,
        type=int
    )

    parser.add_argument(
        '--num_gpus',
        help='Number of GPUs',
        required=False,
        default=1,
        type=int
    )

    parser.add_argument(
        '--eval_delay_secs',
        help='How long to wait before running first evaluation',
        default=1,
        type=int
    )
    parser.add_argument(
        '--min_eval_frequency',
        help='Minimum number of training steps between evaluations',
        default=1,
        type=int
    )

    print(parser)
    args = parser.parse_args()


    # Run the training job
    tf.compat.v1.app.run()

    """""
    tensors_to_log = {'Check_rand':'Check_rand','PRECISION':'PRECISION',

                      'TASK_LOSS_'+IMAGE_CLEF_KEYS.SEVERITY:'TASK_LOSS_'+IMAGE_CLEF_KEYS.SEVERITY,
                      'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: 'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED,
                      'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: 'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED,
                      'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: 'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE,
                      'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: 'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_CALCIFICATION,
                      'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_PLEURISY: 'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_PLEURISY,
                      'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_CAVERNS: 'TASK_LOSS_' + IMAGE_CLEF_KEYS.CTR_CAVERNS,

                      'LOSS_ADD_'+IMAGE_CLEF_KEYS.SEVERITY:'LOSS_ADD_'+IMAGE_CLEF_KEYS.SEVERITY,
                      'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: 'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED,
                      'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: 'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED,
                      'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: 'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE,
                      'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_CALCIFICATION: 'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_CALCIFICATION,
                      'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_PLEURISY: 'LOSS_ADD_' + IMAGE_CLEF_KEYS.CTR_PLEURISY,
                      'LOSS_ADD_'+ IMAGE_CLEF_KEYS.CTR_CAVERNS: 'LOSS_ADD_'+ IMAGE_CLEF_KEYS.CTR_CAVERNS,

                      'log_var_'+ IMAGE_CLEF_KEYS.SEVERITY:'log_var_'+ IMAGE_CLEF_KEYS.SEVERITY,
                      'log_var_' + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED:'log_var_' + IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED,
                      'log_var_' + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED:'log_var_' + IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED,
                      'log_var_' + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE:'log_var_' + IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE,
                      'log_var_' + IMAGE_CLEF_KEYS.CTR_CALCIFICATION:'log_var_' + IMAGE_CLEF_KEYS.CTR_CALCIFICATION,
                      'log_var_' + IMAGE_CLEF_KEYS.CTR_PLEURISY:'log_var_' + IMAGE_CLEF_KEYS.CTR_PLEURISY,
                      'log_var_' + IMAGE_CLEF_KEYS.CTR_CAVERNS:'log_var_' + IMAGE_CLEF_KEYS.CTR_CAVERNS,

                      'REDUCED_LOSS':'REDUCED_LOSS'}
    """