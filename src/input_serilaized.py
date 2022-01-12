#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:58:32 2019

@author: Pedro M. Gordaliza
"""

import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
import glob

#from utils import _float_feature, _int64_feature, _bytes_feature

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



def read_mhd_as_array(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))



def img_vol_to_tfrecord(img_paths, record_path, tfrecords_filename = "created_tf_record", writer = None, chunks = 5):
    """
    img_paths 
    Follow same rules that model.read_and_decode
    """
    
    chunk_size = int(len(img_paths) / float(chunks) )
    chunk_lists = [ img_paths[x:x+chunk_size] for x in range(0,len(img_paths),chunk_size) ]
    
    for i,lst in enumerate(chunk_lists):
        wr_name = os.path.join(record_path,tfrecords_filename+str(i)+'.tfrecords') if chunks > 1 else os.path.join(record_path,lst[0].split(os.sep)[-1][:-4]+'.tfrecords')
        print(wr_name)
        writer = tf.python_io.TFRecordWriter(wr_name)
    
        for j, img_path in  enumerate(lst):
            print (img_path)
            assert(img_path.endswith(".mhd"))
            img_np = read_mhd_as_array(img_path)
        
            name = img_path.split(os.sep)[-1][:-4]
            
            depth, height, width = img_np.shape
            
            example = tf.train.Example(features=tf.train.Features(feature={
            'ID':_bytes_feature(name),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'image_raw': _float_feature(img_np.ravel())}))
        
            writer.write(example.SerializeToString())
    
        writer.close()


def imgpath_to_tf_record(path, record_path):
    img_vol_to_tfrecord([path], record_path, chunks=1)


def decode(path_serialized, return_name = False):
    record_iterator = tf.python_io.tf_record_iterator(path=path_serialized)
    example = tf.train.Example()
    path = next(record_iterator)
    example.ParseFromString(path)
    
    name = (example.features.feature['ID'].bytes_list.value[0])
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    depth = int(example.features.feature['depth'].int64_list.value[0])

    
    img_string = (example.features.feature['image_raw'].float_list.value)
    
    img_1d = np.array(img_string)
    reconstructed_img = img_1d.reshape((depth, height, width))
    
    
    if return_name:
        return reconstructed_img, name
    return reconstructed_img
    
if __name__ == '__main__':
    images_crop = glob.glob('/media/amunoz/anottation_inference/VNet-Tensorflow/resized_cropped_monkeys/*.mhd')
    #img_vol_to_tfrecord(images_crop, '/media/amunoz/anottation_inference/VNet-Tensorflow/serialized_cropped_monkeys/', chunks=4)
    #for img_path in images_crop:
        #imgpath_to_tf_record(img_path, '/media/amunoz/anottation_inference/VNet-Tensorflow/serialized_cropped_monkeys/')
    
    img,name = decode('/media/amunoz/anottation_inference/VNet-Tensorflow/serialized_cropped_monkeys/U12_8.tfrecords', return_name=True)
    
    
    


    