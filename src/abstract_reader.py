from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import os
import pandas as pd
import sys
import traceback
import glob

import SimpleITK as sitk

import tensorflow as tf
from tensorflow.python.lib.io import file_io

from dltk.io.augmentation import (add_gaussian_noise, add_gaussian_offset,
                                  flip, extract_random_example_array, elastic_transform)
from dltk.io.preprocessing import normalise_one_one

from utils import IMAGE_CLEF_KEYS, IMAGE_CLEF_META, EXOTIC_EXAMPLES
from model import Aim


def read_itk_img_decode_for_vnet_out(img_path, from_tensor=True, normalise=True):
    """
    Till now can handle any medical format sorted locally and .mhd files stored in the GCP
    """
    img = sitk.ReadImage(img_path.decode("utf-8") if from_tensor else img_path)
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr.transpose()
    img_arr = normalise_one_one(img_arr) if normalise else img_arr
    img_arr = img_arr[:, :, :, np.newaxis]
    return img_arr


class Data_Workaround(object):

    def __init__(self, mode, reports_path=None, masks_dir=None, normalise_input=True,
                 data_shape=[[124, 124, 64, 1], [10], [124, 124, 64, 1], [8], []]):
        self.mode = mode
        self.reports_path = reports_path
        self.pandas_report = None
        self.normalise_input = normalise_input
        self.data_shape = data_shape
        if self.reports_path is not None:
            with file_io.FileIO(self.reports_path, mode='r') as f:
                self.pandas_report = pd.read_csv(f)

        self.masks_dir = masks_dir

        if self.masks_dir is None and self.pandas_report is None:
            print('Provide masks or reports path to establish an aim')
            sys.exit(1)
        self.aim = Aim.CLASSIFICATION
        if self.masks_dir is not None:
            if self.pandas_report is None:
                self.aim = Aim.SEGMENTATION
            else:
                self.aim = Aim.CLASS_AND_SEG

        print('DATA AIM', self.aim)

    def data_augmenting(self, image, mask=None):
        # TODO try new transformation (elastic, generative..., etc.)
        offset_sig = np.random.rand() * 0.01 if self.normalise_input else np.random.rand() * 0.1
        noise_sig = np.random.rand() * 0.0025 if self.normalise_input else np.random.rand() * 0.025
        flip_ax = np.random.randint(0, 3)
        # print('offset_sig', offset_sig, ' noise_sig', noise_sig, 'flip_ax', flip_ax)
        dice_toss_transforms = np.random.rand(2)
        img = add_gaussian_noise(image, sigma=noise_sig) if dice_toss_transforms[0] > 0.5 else image
        img = add_gaussian_offset(img, sigma=offset_sig) if dice_toss_transforms[1] > 0.5 else img
        # Flip already has a random factor for the transformation!! > 0.5
        images_to_flip = [img, mask] if isinstance(mask, np.ndarray) else [img]  # when mask are not necessary
        flipped_images = flip(images_to_flip, axis=flip_ax)

        # return img and mask flipped when is the case
        if len(flipped_images) > 1:
            return flipped_images[0], flipped_images[1]
        else:
            return flipped_images[0], mask

    def read_itk_img_decode_for_vnet(self, img_path, from_tensor=True, normalise=True):
        """
        Till now can handle any medical format sorted locally and .mhd files stored in the GCP
        """
        img = sitk.ReadImage(img_path.decode("utf-8") if from_tensor else img_path)
        img_arr = sitk.GetArrayFromImage(img)
        img_arr = img_arr.transpose()
        img_arr = normalise_one_one(img_arr) if normalise else img_arr
        return img_arr

    def get_metadata(self, img_path, from_tensor=True):
        if isinstance(self.pandas_report, str):
            with file_io.FileIO(self.pandas_report, mode='r') as f:
                annotatations_pandas = pd.read_csv(f)
        else:
            annotatations_pandas = self.pandas_report

        file_reference = img_path.decode("utf-8") if from_tensor else img_path
        img_fields = file_reference.split(os.sep)[-1]
        name = img_fields + '.gz'
        anotations = annotatations_pandas[annotatations_pandas['Filename'] == name]

        meta_disabilty = anotations[IMAGE_CLEF_META.DISABILITY].values[0]
        meta_relapse = anotations[IMAGE_CLEF_META.RELAPSE].values[0]
        meta_tb_symps = anotations[IMAGE_CLEF_META.TB_SYMPTONS].values[0]
        meta_comorbidity = anotations[IMAGE_CLEF_META.COMORBIDITY].values[0]
        meta_bacilary = anotations[IMAGE_CLEF_META.BACILARY].values[0]
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
        return meta_information

    def get_anotations_labels_decode(self, img_path, from_tensor=True):
        if isinstance(self.pandas_report, str):
            with file_io.FileIO(self.pandas_report, mode='r') as f:
                annotatations_pandas = pd.read_csv(f)
        else:
            annotatations_pandas = self.pandas_report

        file_reference = img_path.decode("utf-8") if from_tensor else img_path
        img_fields = file_reference.split(os.sep)[-1]
        name = img_fields + '.gz'

        anotations = annotatations_pandas[annotatations_pandas['Filename'] == name]
        annotation_severity = anotations[IMAGE_CLEF_KEYS.SEVERITY].values[0]
        annotation_left_lung = anotations[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED].values[0]
        annotation_right_lung = anotations[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED].values[0]
        annotation_lung_capacity = anotations[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE].values[0]
        anotation_calcification = anotations[IMAGE_CLEF_KEYS.CTR_CALCIFICATION].values[0]
        anotation_pleurisy = anotations[IMAGE_CLEF_KEYS.CTR_PLEURISY].values[0]
        anotation_caverns = anotations[IMAGE_CLEF_KEYS.CTR_CAVERNS].values[0]
        anotation_svr_severity = anotations[IMAGE_CLEF_KEYS.SVR_SEVERITY].values[0]
        # "LOW" (scores 4 and 5) and "HIGH" (scores 1, 2 and 3).
        anotation_svr_severity = 1 if anotation_svr_severity == 'LOW' else 0

        y = np.array([annotation_severity,
                      annotation_left_lung,
                      annotation_right_lung,
                      annotation_lung_capacity,
                      anotation_calcification,
                      anotation_pleurisy,
                      anotation_caverns,
                      anotation_svr_severity])

        return y

    def data_workaround(self, img_path):
        """
        :param img_path: :return: the 3D image at img_path. meta_info as vector, see :meth: "get_metadata",
        same for mask and labels. the img path is returned as name
        """
        img = self.read_itk_img_decode_for_vnet(img_path, from_tensor=True, normalise=self.normalise_input)
        name = str(os.path.basename(img_path))
        # Tensorflow and proper none types don't like each other
        meta_info = 'None'
        mask = 'None'
        y = 'None'
        if self.aim != Aim.CLASSIFICATION:
            mask_path = os.path.join(self.masks_dir, os.path.basename(img_path.decode("utf-8")))
            mask = self.read_itk_img_decode_for_vnet(mask_path, from_tensor=False, normalise=False)
            mask = mask.astype(np.float32)
            assert mask.shape == img.shape

        if self.mode == tf.estimator.ModeKeys.TRAIN:
            img, mask = self.data_augmenting(img, mask=mask)

        img = img[:, :, :, np.newaxis]
        img = img.astype(np.float32)

        if self.aim != Aim.CLASSIFICATION:
            mask = mask[:, :, :, np.newaxis]
            mask = mask.astype(np.float32)

        if self.aim != Aim.SEGMENTATION:
            y = self.get_anotations_labels_decode(img_path, from_tensor=True)
            y = y.astype(np.float32)
            meta_info = self.get_metadata(img_path, from_tensor=True)
            meta_info = meta_info.astype(np.float32)

        return img, meta_info, mask, y, name

    def set_shapes_and_dict(self, img, meta_info, mask, y, name):
        img.set_shape(self.data_shape[0])
        if meta_info.dtype != tf.string:
            meta_info.set_shape(self.data_shape[1])
        if mask.dtype != tf.string:
            mask.set_shape(self.data_shape[2])
        if y.dtype != tf.string:
            y.set_shape(self.data_shape[3])
            y = y[:, np.newaxis]
            y = {IMAGE_CLEF_KEYS.SEVERITY: y[0],
                 IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED: y[1],
                 IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED: y[2],
                 IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE: y[3],
                 IMAGE_CLEF_KEYS.CTR_CALCIFICATION: y[4],
                 IMAGE_CLEF_KEYS.CTR_PLEURISY: y[5],
                 IMAGE_CLEF_KEYS.CTR_CAVERNS: y[6],
                 IMAGE_CLEF_KEYS.SVR_SEVERITY: y[7]}
        name.set_shape(self.data_shape[4])
        return {'features': {'x': img, 'meta_info': meta_info}}, {'labels': {'Mask': mask, 'y': y, 'Name': name}}


def set_shapes_and_dict2(img, shapes=[[124, 124, 64, 1]]):
    img.set_shape(shapes[0])
    return {'features': {'x': img}}


# class IteratorInitializerHook(tf.train.SessionRunHook):
class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


class Reader(object):
    """Wrapper for dataset generation given a read function"""

    def __init__(self, read_fn, dtypes):
        """Constructs a Reader instance

        Args:
            read_fn: Input function returning features which is a dictionary of
                string feature name to `Tensor` or `SparseTensor`. If it
                returns a tuple, first item is extracted as features.
                Prediction continues until `input_fn` raises an end-of-input
                exception (`OutOfRangeError` or `StopIteration`).
            dtypes:  A nested structure of tf.DType objects corresponding to
                each component of an element yielded by generator.

        """
        self.dtypes = dtypes

        self.read_fn = read_fn

    def get_inputs(self,
                   file_references,
                   mode,
                   example_shapes=None,
                   shuffle_cache_size=100,
                   batch_size=4,
                   params=None):
        """
        Function to provide the input_fn for a tf.Estimator.

        Args:
            file_references: An array like structure that holds the reference
                to the file to read. It can also be None if not needed.
            mode: A tf.estimator.ModeKeys. It is passed on to `read_fn` to
                trigger specific functions there.
            example_shapes (optional): A nested structure of lists or tuples
                corresponding to the shape of each component of an element
                yielded by generator.
            shuffle_cache_size (int, optional): An `int` determining the
                number of examples that are held in the shuffle queue.
            batch_size (int, optional): An `int` specifying the number of
                examples returned in a batch.
            params (dict, optional): A `dict` passed on to the `read_fn`.

        Returns:
            function: a handle to the `input_fn` to be passed the relevant
                tf estimator functions.
            tf.train.SessionRunHook: A hook to initialize the queue within
                the dataset.
        """
        iterator_initializer_hook = IteratorInitializerHook()
        feats_shapes = example_shapes['features']
        num_parallel_calls = params['num_parallel_calls']
        annot_file = params['reports_path']
        masks_dir = params['masks_dir']
        augment_train = params['augment_train']
        augment_train = tf.estimator.ModeKeys.TRAIN \
            if augment_train and mode == tf.estimator.ModeKeys.TRAIN else tf.estimator.ModeKeys.EVAL
        normalise = params['normalize_input']
        if mode == tf.estimator.ModeKeys.TRAIN:
            repetitions = params['augment_dificult']
            file_references = add_coincideces(file_references, EXOTIC_EXAMPLES,
                                              repeats=repetitions) if repetitions > 0 else file_references
        print('IMAGES_IN', file_references)

        labels_shape = example_shapes['labels']
        feats_types = self.dtypes['features']
        labels_types = self.dtypes['labels']
        data_workd = Data_Workaround(augment_train, reports_path=annot_file, masks_dir=masks_dir,
                                     normalise_input=normalise,
                                     data_shape=[feats_shapes['x'],
                                                 feats_shapes['meta_info'],
                                                 labels_shape['Mask'], labels_shape['y'], labels_shape['Name']])

        def train_inputs():
            # def f():
            #     def clean_ex(ex, compare):
            #         # Clean example dictionary by recursively deleting
            #         # non-relevant entries. However, this does not look into
            #         # dictionaries nested into lists
            #         for k in list(ex.keys()):
            #             if k not in list(compare.keys()):
            #                 del ex[k]
            #             elif isinstance(ex[k], dict) and isinstance(compare[k], dict):
            #                 clean_ex(ex[k], compare[k])
            #             elif (isinstance(ex[k], dict) and not isinstance(compare[k], dict)) or \
            #                     (not isinstance(ex[k], dict) and isinstance(compare[k], dict)):
            #                 raise ValueError('Entries between example and '
            #                                  'dtypes incompatible for key {}'
            #                                  ''.format(k))
            #             elif (isinstance(ex[k], list) and not isinstance(compare[k], list)) or \
            #                     (not isinstance(ex[k], list) and isinstance(compare[k], list)) or \
            #                     (isinstance(ex[k], list) and isinstance(compare[k], list) and not
            #                     len(ex[k]) == len(compare[k])):
            #                 raise ValueError('Entries between example and '
            #                                  'dtypes incompatible for key {}'
            #                                  ''.format(k))
            #         for k in list(compare):
            #             if k not in list(ex.keys()):
            #                 raise ValueError('Key {} not found in ex but is '
            #                                  'present in dtypes. Found keys: '
            #                                  '{}'.format(k, ex.keys()))
            #         return ex
            #
            #     fn = self.read_fn(file_references, mode, params)
            #     # iterate over all entries - this loop is terminated by the
            #     # tf.errors.OutOfRangeError or StopIteration thrown by the
            #     # read_fn
            #     while True:
            #         try:
            #             ex = next(fn)
            #
            #             if ex.get('labels') is None:
            #                 ex['labels'] = None
            #
            #             if not isinstance(ex, dict):
            #                 raise ValueError('The read_fn has to return '
            #                                  'dictionaries')
            #
            #             ex = clean_ex(ex, self.dtypes)
            #             yield ex
            #         except (tf.errors.OutOfRangeError, StopIteration):
            #             raise
            #         except Exception as e:
            #             print('got error `{} from `_read_sample`:'.format(e))
            #             print(traceback.format_exc())
            #             raise

            dataset = tf.data.Dataset.list_files(file_references)
            dataset = dataset.shuffle(shuffle_cache_size) if mode == tf.estimator.ModeKeys.TRAIN else dataset
            dataset = dataset.map(lambda filename: tuple(tf.py_func(data_workd.data_workaround, [filename],
                                                                    [feats_types['x'], feats_types['meta_info'],
                                                                     labels_types['Mask'], labels_types['y'],
                                                                     labels_types['Name']])),
                                  num_parallel_calls=num_parallel_calls)

            def work_a(img, meta_info, mask, y, name):
                return data_workd.set_shapes_and_dict(img, meta_info, mask, y, name)

            dataset = dataset.map(work_a, num_parallel_calls=num_parallel_calls)
            # TODO shuffle --> batch --> repeat
            dataset = dataset.batch(batch_size)
            dataset = dataset.repeat(None)
            dataset = dataset.prefetch(batch_size*4)  # TODO OJO que esto puede reventar
            return dataset

        return train_inputs  # , iterator_initializer_hook

    def serving_input_receiver_fn(self, placeholder_shapes):
        """Build the serving inputs.

        Args:
            placeholder_shapes: A nested structure of lists or tuples
                corresponding to the shape of each component of the feature
                elements yieled by the read_fn.

        Returns:
            function: A function to be passed to the tf.estimator.Estimator
            instance when exporting a saved model with estimator.export_savedmodel.
        """

        def f():
            inputs = {k: tf.placeholder(
                shape=[None] + list(placeholder_shapes['features'][k]),
                dtype=self.dtypes['features'][k]) for k in list(self.dtypes['features'].keys())}

            print("INPUTS", inputs)

            return tf.estimator.export.ServingInputReceiver(inputs, inputs)

        print("PLACE_HOLDER", placeholder_shapes)
        return f


def my_service(file_references, report, normalise):
    dw = Data_Workaround(mode=tf.estimator.ModeKeys.PREDICT, reports_path=report)
    for file_ref in file_references:
        img = dw.read_itk_img_decode_for_vnet(file_ref, from_tensor=False, normalise=normalise)
        img = img[:, :, :, np.newaxis]
        meta_info = dw.get_metadata(file_ref, from_tensor=False)
        # img = read_itk_img_decode_for_vnet_out(file_ref, from_tensor=False)
        yield img, meta_info, file_ref.split(os.sep)[-1].split('.')[0]


def add_coincideces(a_list, id_list, repeats=1):
    coincidences = []
    for id_l in id_list:
        for a_l in a_list:
            if id_l in a_l:
                coincidences = coincidences + [a_l] * repeats
    return a_list + coincidences


if __name__ == '__main__':
    files = glob.glob(
        '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/Training_SET_MASKS_Non_ISO_2/*.nii')
    ids = ['CTR_TRN_004']
    print(files)
    print(add_coincideces(files, ids))
