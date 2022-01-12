#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 14:29:24 2019

@author: pedro
"""
import SimpleITK as sitk
import os
from glob import glob

from utils import center_images, resample_image
from rib_cage import crop_images_to_lung_mask

ORIGINAL_DATA_PATH = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/'


ref_path = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_2_of_2/CTR_TRN_108.nii.gz'
similar_image_path = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_2_of_2/CTR_TRN_115.nii.gz'
low_res_path =  '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_1_of_2/CTR_TRN_072.nii.gz'
high_res_path = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_1_of_2/CTR_TRN_079.nii.gz'
big_image_path = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_1_of_2/CTR_TRN_087.nii.gz'

MASKS_PATH = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_Modified_Masks/'
extension = '.nii.gz'
images_trn_1 = glob(os.path.join(ORIGINAL_DATA_PATH, 'TrainingSet_1_of_2/')+'*'+extension)
images_trn_2 = glob(os.path.join(ORIGINAL_DATA_PATH, 'TrainingSet_2_of_2/')+'*'+extension)
images_trn = images_trn_1 + images_trn_2

ref_image = sitk.ReadImage(ref_path)
name_ref = ref_path.split('/')[-1]
ref_mask = sitk.ReadImage(os.path.join(MASKS_PATH, name_ref.split('.')[0]+'_1.nii'))
ref_mask.CopyInformation(ref_image)
ref_crop_img, ref_crop_msk = crop_images_to_lung_mask(ref_image, ref_mask, square_images=False)


sample_images = [similar_image_path, low_res_path, high_res_path, big_image_path]


for img in images_trn:
    print(img)
    name = img.split(os.sep)[-1]
    image = sitk.ReadImage(img)
    mask = sitk.ReadImage(os.path.join(MASKS_PATH, name.split('.')[0]+'_1.nii'))
    mask.CopyInformation(image)
    #mask = sitk.Cast(mask, sitk.sitkUInt8)
    name = name[:-3]
    
    c = sitk.MinimumMaximumImageFilter()
    c.Execute(image)
    center_img = center_images(ref_image, image, defaul_value=c.GetMinimum())
    center_mask = center_images(ref_image, mask, label=True, defaul_value=0)
    crop_img, crop_msk = crop_images_to_lung_mask(center_img, center_mask)
    c.Execute(crop_img)
    resampled_cropped_img = resample_image(crop_img,out_spacing=(1.7, 1.7, 4.7), out_size=(196,144,72), deafault_value=c.GetMinimum() )
    resampled_cropped_msk = resample_image(crop_msk,out_spacing=(1.7, 1.7, 4.7), out_size=(196, 144,72),is_label=True, deafault_value=0)
    
    
    
    sitk.WriteImage(resampled_cropped_img, '/home/pedro/Documents/projects/bodyct-tuberculosis-multitask/Training_SET_Non_ISO_2/'+name)
    sitk.WriteImage(resampled_cropped_msk, '/home/pedro/Documents/projects/bodyct-tuberculosis-multitask/Training_SET_MASKS_Non_ISO_2/'+name)
