#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:46:33 2018

@author: Pedro M. Gordaliza
"""
import SimpleITK
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi
import os

import glob

def multi_label_mask(itk_image, labels):
    img_out = SimpleITK.Image(itk_image.GetSize(), SimpleITK.sitkUInt8)
    img_out.CopyInformation(itk_image)
    for l in labels:
        img_out+= itk_image == l
    return img_out

def distance_objects(itk_image, image_center = None):
    size = np.array(itk_image.GetSize())
    image_center = itk_image.TransformIndexToPhysicalPoint( [ int(s/2) for s in size ] )  if image_center is None else image_center
    shape_stats = SimpleITK.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(itk_image)
    return np.array(shape_stats.GetLabels()),[euclidean(shape_stats.GetCentroid(label), image_center) for label in shape_stats.GetLabels() ]

def get_scapula_labels(bone_image, ref_slice = None, num_s = 2):
    if ref_slice is not None:
        ref_slice = ref_slice
        bone_image = bone_image[:,:,ref_slice]
    labels,distances = distance_objects(bone_image)

    return labels[np.argsort(distances)[-num_s:]].reshape(num_s,1) if len(distances) > 1 else labels.reshape(1,1)

def get_CT_bones_rude(itk_image, n_th = 3, bone_limit = [500,1200], size_limit = 1.0, ref_slice = -1):
    '''
    Basically separates the volume  hypothesizing trimodal histogram (bone/background/tissue) to obtain the bone (higher histogrmam)
    Next, get the connected bones, filtering outliers in intensity, distance and size to get and remove the scapulas
    '''
    
    otsu = SimpleITK.OtsuMultipleThresholds(itk_image, numberOfThresholds = n_th)
    intensity_stats = SimpleITK.LabelIntensityStatisticsImageFilter()
    intensity_stats.Execute(otsu, itk_image)
    SimpleITK.WriteImage(otsu, '/tmp/otsu.nii')

    labels_median = [intensity_stats.GetMedian(l) for l in intensity_stats.GetLabels() ]
    sorted_labels_indx = np.argsort(labels_median)

    bone_label = sorted_labels_indx[-1] + 1 #Max intensity is bone
    print('bone_labels',bone_label)
    
    lung_label = sorted_labels_indx[0] + 1 #After backgorund lowe intensity should be lungs. Always under the trimodal hypotesis
    print('Lung label' ,lung_label)

    bone_mask = otsu == bone_label
    SimpleITK.WriteImage(bone_mask, '/tmp/bone_mask_pre.nii')
    bone_mask = SimpleITK.Median(bone_mask, [2,2,2])
    SimpleITK.WriteImage(bone_mask, '/tmp/bone_mask.nii')
    connected_bones = SimpleITK.ConnectedComponent(bone_mask)
    SimpleITK.WriteImage(connected_bones, '/tmp/connected_bone.nii')
    
    lung_mask = otsu == lung_label
    lung_mask = SimpleITK.Median(lung_mask, [4,4,4])
    connected_lung = SimpleITK.ConnectedComponent(lung_mask)
    SimpleITK.WriteImage(connected_lung, '/tmp/connected_lung.nii')
    #TODO. Now is Ad-hoc for simpel images. Should be probabilistic
    intensity_stats.Execute(connected_lung, itk_image)
    lungs_sizes = [intensity_stats.GetPhysicalSize(i) for i in intensity_stats.GetLabels()]
    lung_sizes_indx_sort = np.argsort(lungs_sizes)
    print('lung label', intensity_stats.GetLabels()[lung_sizes_indx_sort[-2]] )
    lung_mask = connected_lung == intensity_stats.GetLabels()[lung_sizes_indx_sort[-2]] 
    
    intensity_stats.Execute(connected_bones, itk_image)
    labels = np.array(intensity_stats.GetLabels())

    intensities = np.array([intensity_stats.GetMedian( int(l) ) for l in labels ])
    labels = labels [(intensities > bone_limit[0]) * (intensities < bone_limit[1])]

    sizes = np.array([intensity_stats.GetPhysicalSize(int(l) ) for l in labels])
    
    labels = labels[ sizes > size_limit  ]

    
    lowers = np.array([intensity_stats.GetCentroid(int(l) )[1] for l in labels])
    lim_y = itk_image.TransformIndexToPhysicalPoint(itk_image.GetSize())[1]
    labels = labels[lowers < (lim_y - 9)]

    
    
    connected_bones = SimpleITK.Mask(connected_bones, multi_label_mask(connected_bones, labels))
    SimpleITK.WriteImage(connected_bones, '/tmp/connected_bone_to_scap.nii')
    scp_labels = get_scapula_labels(connected_bones, ref_slice = ref_slice)
    print (scp_labels, type(scp_labels))
    labels = labels[ np.prod(labels != scp_labels, axis = 0, dtype = np.bool) ]
    no_scp_bones = multi_label_mask(connected_bones, labels)
    return otsu, bone_mask, SimpleITK.Mask(itk_image, bone_mask), connected_bones,SimpleITK.Mask(connected_bones, no_scp_bones), SimpleITK.Mask(itk_image, no_scp_bones), no_scp_bones, lung_mask, connected_lung


class MASK_DOWNSAMPLING():
    CONTOUR = SimpleITK.BinaryContourImageFilter()
    THINNING = SimpleITK.BinaryThinningImageFilter()

def get_rib_cage_convex_hull(rib_cage_mask, downsamplig = MASK_DOWNSAMPLING.THINNING):
    rib_cage_mask = downsamplig.Execute(rib_cage_mask) if downsamplig is not None else rib_cage_mask
    mask_array = SimpleITK.GetArrayFromImage(rib_cage_mask)
    points = np.stack([indx for indx in np.where(mask_array)], axis = 1)
    return points,Voronoi(points) #ConvexHull(points)

def get_bounding_box(original_image, rib_cage_mask, lung_mask,  include_bones = True):
    if original_image.GetSize() != rib_cage_mask.GetSize():
        print('Dimensions must be equal:',original_image.GetSize(), rib_cage_mask.GetSize())
    h,w,d = original_image.GetSize()
    rib_cage_mask = rib_cage_mask > 0 #Just in case
    box_filter = SimpleITK.LabelShapeStatisticsImageFilter()
    box_filter.Execute(rib_cage_mask)
    x_rc_1,y_rc_1,z_rc_1,dx_rc,dy_rc,dz_rc = box_filter.GetBoundingBox(1)
    x_rc_2 = x_rc_1 + dx_rc; y_rc_2 = y_rc_1 + dy_rc; z_rc_2 = z_rc_1 + dz_rc #Rib cage points bounding box
    
    box_filter.Execute(lung_mask)
    x_lm_1,y_lm_1,z_lm_1,dx_lm,dy_lm,dz_lm = box_filter.GetBoundingBox(1)
    x_lm_2 = x_lm_1 + dx_lm; y_lm_2 = y_lm_1 + dy_lm; z_lm_2 = z_lm_1 + dz_lm
    

    print('ribcg points',x_rc_1,y_rc_1,z_rc_1,x_rc_2,y_rc_2,z_rc_2)
    print('lungs points',x_lm_1,y_lm_1,z_lm_1,x_lm_2,y_lm_2,z_lm_2 )
    x1 = min(x_rc_1, x_lm_1)
    y1 = min(y_rc_1, y_lm_1)
    z1 = min(z_rc_1, z_lm_1)
    x2 = max(x_rc_2, x_lm_2)
    y2 = max(y_rc_2, y_lm_2)
    z2 = max(z_rc_2, z_lm_2)
    print(x1,y1,z1,x2,y2,z2)
    
    
    original_image = SimpleITK.Mask(original_image, rib_cage_mask < 1) if include_bones is False else original_image
    print([h - x2, w - y2, d - z2])
    return SimpleITK.Crop(original_image, [x1,y1,z1], [h - x2, w - y2, d - z2])

def crop_images_to_lung_mask(original_image, original_lung_mask, mask_label=None, square_images = False):
    if original_image.GetSize() != original_lung_mask.GetSize():
        print('Dimensions must be equal:',original_image.GetSize(), original_lung_mask.GetSize())
    h,w,d = original_image.GetSize()
    original_lung_mask = original_lung_mask > 0 if mask_label is None else original_lung_mask == mask_label
    
    box_filter = SimpleITK.LabelShapeStatisticsImageFilter()
    box_filter.Execute(original_lung_mask)
    x_rc_1,y_rc_1,z_rc_1,dx_rc,dy_rc,dz_rc = box_filter.GetBoundingBox(1)
    x_rc_2 = x_rc_1 + dx_rc; y_rc_2 = y_rc_1 + dy_rc; z_rc_2 = z_rc_1 + dz_rc #points bounding
    
    h_crop = h - x_rc_2
    w_crop = w - y_rc_2
    
    if square_images:
        n_points = max(dx_rc, dy_rc)
        x_rc_2 = x_rc_1 + n_points; y_rc_2 = y_rc_1 + n_points;
        h_crop = h - x_rc_2
        w_crop = w - y_rc_2
    
    return SimpleITK.Crop(original_image, [x_rc_1,y_rc_1,z_rc_1], [h_crop, w_crop, d - z_rc_2 ]), SimpleITK.Crop(original_lung_mask, [x_rc_1,y_rc_1,z_rc_1], [h_crop, w_crop, d - z_rc_2])

def crop_sample(images_paths, output_path_img, output_path_msk, mask_label = 1):
    for img_path in images_paths:
        name = img_path.split('/')[-1]
        name_ext = name.split('.')[0]
        mask_path = os.path.join(MASKS_PATH, name_ext+'_1.nii')
        print(name, mask_path)
        image = SimpleITK.ReadImage(img_path)
        mask = SimpleITK.ReadImage(mask_path)

        image_crop, mask_crop = crop_images_to_lung_mask(image, mask, mask_label=mask_label)
        # image_crop = SimpleITK.Cast(image_crop, SimpleITK.sitkInt16)
        SimpleITK.WriteImage(image_crop, output_path_img + name_ext + '.nii')
        SimpleITK.WriteImage(mask_crop, output_path_msk + name_ext + '.nii')


if __name__ == "__main__":
    IMAGES_PATH = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019'
    lobes = [1,2]
    sets = [True, False]
    for training_set in sets:
        set_name = 'TrainingSet_Cropped' if training_set else 'TestSet_Cropped'
        MASKS_PATH = os.path.join(IMAGES_PATH, 'TrainingSet_Modified_Masks') if training_set else os.path.join(IMAGES_PATH, 'TestSet_Modified_Masks')
        for lobe in lobes:
            lobe_antomical = 'Left' if lobe == 1 else 'Right'
            images2 = glob.glob(os.path.join(IMAGES_PATH,'TrainingSet_2_of_2/*.gz')) if training_set else glob.glob(os.path.join(IMAGES_PATH,'TestSet/*.gz'))
            images1 = glob.glob(os.path.join(IMAGES_PATH,'TrainingSet_1_of_2/*.gz'))
            images = images2 + images1 if training_set else images2
            folder_name = set_name + '_' + lobe_antomical
            path_to_save_images = os.path.join('/tmp/', folder_name+os.sep)
            os.mkdir(path_to_save_images)
            path_to_save_masks = os.path.join('/tmp/', folder_name+'_Masks'+os.sep)
            os.makedirs(path_to_save_masks)
            crop_sample(images, path_to_save_images, path_to_save_masks, mask_label=lobe)


#    for img_path in images:      
#        name = img_path.split('/')[-1]
#        name_ext = name.split('.')[0]
#        mask_path = os.path.join(MASKS_PATH, name)
#        print(name, mask_path)
#        image = SimpleITK.ReadImage(img_path)
#        mask = SimpleITK.ReadImage(mask_path)
#        try:
#            print('at 3 th')
#            a = get_CT_bones_rude(image, n_th=3, bone_limit=[200,1300])
#        except:
#            print('at 4 th')
#            a = get_CT_bones_rude(image, n_th=4, bone_limit=[200,1300]) #cheap workaround
#            continue
#        
#        #SimpleITK.WriteImage(a[0],'/tmp/otsu.mhd')
#        #SimpleITK.WriteImage(a[1],'/tmp/bone_mask.mhd')
#        #SimpleITK.WriteImage(a[2],'/tmp/bone.mhd')
#        #SimpleITK.WriteImage(a[3],'/tmp/connected_bone_'+name)
#        #SimpleITK.WriteImage(a[4],'/tmp/connected_bone_no_scp.mhd')
#        #SimpleITK.WriteImage(a[5],'/tmp/bones_no_scp.mhd')
#        #SimpleITK.WriteImage(a[6],'/tmp/rc_'+name)
#        #SimpleITK.WriteImage(a[7],'/tmp/lung_'+name)
#        SimpleITK.WriteImage(get_bounding_box(image, a[6], a[7]), '/tmp/TrainingSet_Croped_Raw/'+name_ext+'.nii')
#        SimpleITK.WriteImage(get_bounding_box(mask, a[6], a[7]), '/tmp/TrainingSet_Croped_Masks_Raw/'+name_ext+'.nii')
