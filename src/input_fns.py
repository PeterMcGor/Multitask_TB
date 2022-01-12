#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:47:57 2019

@author: Pedro M. Gordaliza
"""
import numpy as np

import tensorflow as tf
from utils import read_fn_lbels_light
import SimpleITK as sitk
from NiftiDataset import NiftiDataset
#
#def input_function(filenames, mode, patch_size, batch_size, buffer_size, valid_id,
#						pred_id, overlap_step, num_epochs=1, num_parallel_calls=1):


def nifty_input_function(train_data_dir):
    trainTransforms = [
                #NiftiDataset.StatisticalNormalization(2.5),
                # NiftiDataset.Normalization(),
                #NiftiDataset.Resample((0.45,0.45,0.45)),
                #NiftiDataset.Padding((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer)),
                #NiftiDataset.RandomCrop((FLAGS.patch_size, FLAGS.patch_size, FLAGS.patch_layer),FLAGS.drop_ratio,FLAGS.min_pixel),
                #NiftiDataset.RandomNoise()
                ]

    TrainDataset = NiftiDataset(
                data_dir=train_data_dir,
                #image_filename=FLAGS.image_filename,
                #label_filename=FLAGS.label_filename,
                transforms=trainTransforms,
                train=True
                )
            
    trainDataset = TrainDataset.get_classification_dataset()
    trainDataset = trainDataset.shuffle(buffer_size=5)
    trainDataset = trainDataset.batch(8)
    
    #train_iterator = trainDataset.make_initializable_iterator()
    train_iterator = trainDataset.make_one_shot_iterator()
    next_element_train = train_iterator.get_next()
    print('NEXT ELEMENT' ,next_element_train)
    return {'x': next_element_train[0]}  , {'y': next_element_train[1], 'Name': next_element_train[2]}
    


def input_function(filenames, num_parallel_calls=1, batch_size=5, buffer_size=10):
    def workaround(dummy1, dummy2):
        print('Dummy', dummy1)
        return read_fn_lbels_light(filenames[dummy2] )
 
    with tf.name_scope('input'):
        indx_list = np.array(range(0, len(filenames) )) 
        print('INDICES',tf.constant(filenames, dtype = tf.string) )
        #feeder = tf.constant(filenames, dtype = tf.string, shape=(10,1) ) 
        #print('INDICES', tf.Session().run( feeder ) )
        #dataset = tf.data.Dataset.from_tensor_slices( feeder )
        
        dataset = tf.data.Dataset.list_files('../../TrainingSet_2mm/*.nii')
        dataset = dataset.map(lambda path: tf.py_function(read_fn_lbels_light,[path], [tf.float32, tf.float32, tf.string]), num_parallel_calls=num_parallel_calls)
        return dataset
#        dataset = dataset.prefetch(buffer_size=batch_size)
#        dataset = dataset.shuffle(buffer_size=buffer_size)
#        dataset = dataset.repeat(None)
#        #dataset = dataset.map(lambda image_path: tuple(tf.py_function(read_fn_lbels_light, [image_path], [tf.float32,tf.float32,tf.string])), num_parallel_calls=num_parallel_calls)
#        
#        dataset = dataset.batch(batch_size)
#        dataset = dataset.prefetch(1)
#        iterator = dataset.make_one_shot_iterator()
#        #iterator = dataset.make_initializable_iterator()
#        
#        out = iterator.get_next()
#        print("OUT",out)
#        return {'x': out[0]}  , {'y': out[1], 'Name': out[2]}
        #return out['features'], out['labels']
    

#		if mode == 'train':
#			# Shuffle the records. Note that we shuffle before repeating to ensure
#			# that the shuffling respects epoch boundaries.
#			dataset = dataset.shuffle(buffer_size=buffer_size)
#
#		# If we are training over multiple epochs before evaluating, repeat the
#		# dataset for the appropriate number of epochs.
#		dataset = dataset.repeat(num_epochs)
#
#		if mode ==  tf.estimator.ModeKeys.TRAIN:
#			dataset = dataset.map(read_fn_lbels_light, num_parallel_calls=num_parallel_calls)
#			#dataset = dataset.map(data_augmenting, num_parallel_calls=num_parallel_calls)
#		elif mode ==  tf.estimator.ModeKeys.EVAL:
#			dataset = dataset.map(read_itk_img_for_vnet, num_parallel_calls=num_parallel_calls)
#		elif mode == tf.estimator.ModeKeys.PREDICT:
#			dataset = dataset.map(read_itk_img_for_vnet, num_parallel_calls=num_parallel_calls)
#
#		#dataset = dataset.map(normalize_image, num_parallel_calls=num_parallel_calls)
#
#		dataset = dataset.batch(batch_size)
#
#		# Operations between the final prefetch and the get_next call to the iterator
#		# will happen synchronously during run time. We prefetch here again to
#		# background all of the above processing work and keep it out of the
#		# critical training path.
#		dataset = dataset.prefetch(1)
#
#		iterator = dataset.make_one_shot_iterator()
#		features, label = iterator.get_next()
#
#         return features, label