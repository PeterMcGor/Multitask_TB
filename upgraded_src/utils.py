#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:35:30 2019

@author: Pedro M. Gordaliza
"""


import SimpleITK as sitk
import os
from glob import glob
import numpy as np
import pandas as pd
import sys
from multiprocessing import Pool
import random


import tensorflow as tf

from dltk.io.augmentation import (add_gaussian_noise, add_gaussian_offset,
                                  flip, extract_random_example_array,
                                  elastic_transform)
from dltk.io.preprocessing import normalise_one_one

from mhd_utils import load_raw_data_with_mhd
from input_serilaized import decode as dec_serial
from rib_cage import crop_images_to_lung_mask



from tensorflow.python.lib.io import file_io

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)


ORIGINAL_DATA_PATH = '/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/'

EXOTIC_EXAMPLES = ["CTR_TRN_001","CTR_TRN_004","CTR_TRN_005","CTR_TRN_013","CTR_TRN_017","CTR_TRN_018","CTR_TRN_020",
                   "CTR_TRN_026","CTR_TRN_033","CTR_TRN_040","CTR_TRN_042","CTR_TRN_045","CTR_TRN_046","CTR_TRN_050",
                   "CTR_TRN_057","CTR_TRN_062","CTR_TRN_070","CTR_TRN_071","CTR_TRN_074","CTR_TRN_076",
                   "CTR_TRN_082","CTR_TRN_086","CTR_TRN_091","CTR_TRN_099","CTR_TRN_114","CTR_TRN_128", "CTR_TRN_122",
                   "CTR_TRN_164", "CTR_TRN_183","CTR_TRN_193","CTR_TRN_197","CTR_TRN_203","CTR_TRN_205","CTR_TRN_214",
                   "CTR_TRN_217"]

class LABELS_KEYS:
    RU_NODS = 'Nodules at the rigth upper lobe'
    RU_CAVS = 'Binary for existance of cavitations at the rigth upper lobe'
    RU_CONS = 'Binary for existance of consolidations at the rigth upper lobe'
    RU_CONG = 'Binary for existance of conglomerations at the rigth upper lobe'
    RU_TBDS = 'Binary for existance of trees in bud at the rigth upper lobe'
    
    RM_NODS = 'Nodules at the rigth medium lobe'
    RM_CAVS = 'Binary for existance of cavitations at the rigth medium lobe'
    RM_CONS = 'Binary for existance of consolidations at the rigth medium lobe'
    RM_CONG = 'Binary for existance of conglomerations at the rigth medium lobe'
    RM_TBDS = 'Binary for existance of trees in bud at the rigth medium lobe'
    
    RL_NODS = 'Nodules at the rigth low lobe'
    RL_CAVS = 'Binary for existance of cavitations at the rigth low lobe'
    RL_CONS = 'Binary for existance of consolidations at the rigth low lobe'
    RL_CONG = 'Binary for existance of conglomerations at the rigth low lobe'
    RL_TBDS = 'Binary for existance of trees in bud at the rigth low lobe'
    
    LU_NODS = 'Nodules at the left upper lobe'
    LU_CAVS = 'Binary for existance of cavitations at the left upper lobe'
    LU_CONS = 'Binary for existance of consolidations at the left upper lobe'
    LU_CONG = 'Binary for existance of conglomerations at the left upper lobe'
    LU_TBDS = 'Binary for existance of trees in bud at the left upper lobe'
    
    LL_NODS = 'Nodules at the left low lobe'
    LL_CAVS = 'Binary for existance of cavitations at the left low lobe'
    LL_CONS = 'Binary for existance of consolidations at the left low lobe'
    LL_CONG = 'Binary for existance of conglomerations at the left low lobe'
    LL_TBDS = 'Binary for existance of trees in bud at the left low lobe'


class IMAGE_CLEF_KEYS:
    SEVERITY = "md_Severity"  
    SVR_SEVERITY = 'SVR_Severity'
    CTR_LEFT_LUNG_AFFECTED = "CTR_LeftLungAffected"
    CTR_RIGHT_LUNG_AFFECTED = "CTR_RightLungAffected"
    CTR_LUNG_CAPACITY_DECREASE = "CTR_LungCapacityDecrease"
    CTR_CALCIFICATION = "CTR_Calcification"
    CTR_PLEURISY = "CTR_Pleurisy"
    CTR_CAVERNS = "CTR_Caverns"

    @staticmethod
    def keys_as_list(just_binary=False):
        manifestations = [IMAGE_CLEF_KEYS.SEVERITY, IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED,
                          IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED, IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE,
                          IMAGE_CLEF_KEYS.CTR_CALCIFICATION, IMAGE_CLEF_KEYS.CTR_PLEURISY, IMAGE_CLEF_KEYS.CTR_CAVERNS]
        if just_binary:
            manifestations.remove(IMAGE_CLEF_KEYS.SEVERITY)

        return manifestations



class IMAGE_CLEF_META:
    DISABILITY = "md_Disability"
    RELAPSE = "md_Relapse"
    TB_SYMPTONS = "md_SymptomsOfTB"
    COMORBIDITY = "md_Comorbidity"
    BACILARY = "md_Bacillary"
    DRUG_RESISTANT = "md_DrugResistance"
    HIGHER_EDUCATION = "md_HigherEducation"
    EX_PRISONER = "md_ExPrisoner"
    ALCOHOLIC = "md_Alcoholic"
    SMOKER = "md_Smoking"

    @staticmethod
    def keys_as_list():
        return [IMAGE_CLEF_META.DISABILITY, IMAGE_CLEF_META.RELAPSE, IMAGE_CLEF_META.TB_SYMPTONS,
                IMAGE_CLEF_META.COMORBIDITY, IMAGE_CLEF_META.BACILARY, IMAGE_CLEF_META.DRUG_RESISTANT,
                IMAGE_CLEF_META.HIGHER_EDUCATION, IMAGE_CLEF_META.EX_PRISONER, IMAGE_CLEF_META.ALCOHOLIC,
                IMAGE_CLEF_META.SMOKER]

_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}




def get_image_dimension(img_path):
    return sitk.ReadImage(img_path).GetSize()
  
def resample_sitk_image(sitk_image, spacing=None, interpolator=None,
                        fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    spacing : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not spacing:
        min_spacing = orig_spacing.min()
        new_spacing = [min_spacing]*num_dim
    else:
        new_spacing = [float(s) for s in spacing]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_size = orig_size*(orig_spacing/new_spacing)
    new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_size = [int(s) for s in new_size] #  SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(sitk_image,
                                                   new_size,
                                                   sitk.Transform(),
                                                   sitk_interpolator,
                                                   orig_origin,
                                                   new_spacing,
                                                   orig_direction,
                                                   fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image

def resample_sitk_image_size(sitk_image, size=None, interpolator=None, fill_value=0):
    """Resamples an ITK image to a new grid. If no spacing is given,
    the resampling is done isotropically to the smallest value in the current
    spacing. This is usually the in-plane resolution. If not given, the
    interpolation is derived from the input data type. Binary input
    (e.g., masks) are resampled with nearest neighbors, otherwise linear
    interpolation is chosen.
    Parameters
    ----------
    sitk_image : SimpleITK image or str
      Either a SimpleITK image or a path to a SimpleITK readable file.
    size : tuple
      Tuple of integers
    interpolator : str
      Either `nearest`, `linear` or None.
    fill_value : int
    Returns
    -------
    SimpleITK image.
    """

    if isinstance(sitk_image, str):
        sitk_image = sitk.ReadImage(sitk_image)
    num_dim = sitk_image.GetDimension()

    if not interpolator:
        interpolator = 'linear'
        pixelid = sitk_image.GetPixelIDValue()

        if pixelid not in [1, 2, 4]:
            raise NotImplementedError(
                'Set `interpolator` manually, '
                'can only infer for 8-bit unsigned or 16, 32-bit signed integers')
        if pixelid == 1: #  8-bit unsigned int
            interpolator = 'nearest'

    orig_pixelid = sitk_image.GetPixelIDValue()
    orig_origin = sitk_image.GetOrigin()
    orig_direction = sitk_image.GetDirection()
    orig_spacing = np.array(sitk_image.GetSpacing())
    orig_size = np.array(sitk_image.GetSize(), dtype=np.int)

    if not size:
        min_size = orig_size.min()
        new_size = [min_size]*num_dim
    else:
        new_size = [s for s in size]

    assert interpolator in _SITK_INTERPOLATOR_DICT.keys(),\
        '`interpolator` should be one of {}'.format(_SITK_INTERPOLATOR_DICT.keys())

    sitk_interpolator = _SITK_INTERPOLATOR_DICT[interpolator]

    new_spacing = orig_size*(orig_spacing/new_size)
    #new_size = np.ceil(new_size).astype(np.int) #  Image dimensions are in integers
    new_spacing = [s for s in new_spacing] #  SimpleITK expects lists, not ndarrays

    resample_filter = sitk.ResampleImageFilter()
    
    print('new size', new_size, 'new spacing', new_spacing)

    resampled_sitk_image = resample_filter.Execute(sitk_image, new_size,
                                                   sitk.Transform(), 
                                                   sitk_interpolator, 
                                                   orig_origin, new_spacing,
                                                   orig_direction, fill_value,
                                                   orig_pixelid)

    return resampled_sitk_image


def read_itk_img_for_vnet(img_path):
    """
    Till now can handle any medical format sotred locally and .mhd files stored in the GCP
    """
    image_in_GCP_bucket = img_path.startswith('gs://')
    if image_in_GCP_bucket:
        if img_path.endswith('.mhd'):
            img_arr = load_raw_data_with_mhd(img_path)[0]
        else:
            print('There is not implemented reader for file stored in GCP with the extension', os.path.splitext()[1])
            sys.exit(1)
    else:        
        img = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(img)
        
    return img_arr.transpose()

def read_itk_img_mask_as_np(img_path, msk_path = None):
    """
    When msk_path is None. The function looks for the end at the name for '_img' and '_msk'
    for the image and the mask.
    """
    msk_path = img_path.replace('_img', '_msk') if msk_path is None else msk_path
    return read_itk_img_for_vnet(img_path), read_itk_img_for_vnet(msk_path) #return images as [w,h,d] in np forma

def read_mask_from_image_as_np(img_path, masks_dir):
  """
  reader for mask and files with equal names at different folders
  """
  msk_path = os.path.join( masks_dir,os.path.basename(img_path))
  print('read_mask_from_image_as_np' ,img_path,msk_path)
  return read_itk_img_for_vnet(img_path), read_itk_img_for_vnet(msk_path) #return images as [w,h,d] in np form
  

def read_and_decode(img_path, msk_path = None):
    """
    When msk_path is None. The function looks for the end at the name for '_img' and '_msk'
    for the image and the mask.
    """
    msk_path = img_path.replace('_img', '_msk') if msk_path is None else msk_path
    print('path', img_path)
    img = sitk.ReadImage(img_path)
    msk = sitk.ReadImage(msk_path)
    img_arr = sitk.GetArrayFromImage(img)
    msk_arr = sitk.GetArrayFromImage(msk)
    img = img_arr.transpose()[:,:,:,np.newaxis]
    msk = msk_arr.transpose()[:,:,:,np.newaxis]

    return img, msk


def anootation_to_labels(pandas_df, subject,week):
    assert type(pandas_df) is pd.DataFrame
    
    s_w_df = pandas_df[pandas_df['Week'] == int(week)][pandas_df['ID'] == subject]
    
    return {LABELS_KEYS.RU_NODS:int(s_w_df['PD_L_LOB_RU_Lesion_Count'].values[0]),
            LABELS_KEYS.RU_CAVS:int(s_w_df['PD_L_LOB_RU_Cavitation'].values[0]), 
            LABELS_KEYS.RU_CONS:int(s_w_df['PD_L_LOB_RU_Consolidation'].values[0]), 
            LABELS_KEYS.RU_CONG:int(s_w_df['PD_L_LOB_RU_Conglomeration'].values[0]),
            LABELS_KEYS.RU_TBDS:int(s_w_df['PD_L_LOB_RU_Tree_in_Bud'].values[0]),
            
            LABELS_KEYS.RM_NODS:int(s_w_df['PD_L_LOB_RM_Lesion_Count'].values[0]),
            LABELS_KEYS.RM_CAVS:int(s_w_df['PD_L_LOB_RM_Cavitation'].values[0]), 
            LABELS_KEYS.RM_CONS:int(s_w_df['PD_L_LOB_RM_Consolidation'].values[0]), 
            LABELS_KEYS.RM_CONG:int(s_w_df['PD_L_LOB_RM_Conglomeration'].values[0]),
            LABELS_KEYS.RM_TBDS:int(s_w_df['PD_L_LOB_RM_Tree_in_Bud'].values[0]),
            
            LABELS_KEYS.RL_NODS:int(s_w_df['PD_L_LOB_RL_Lesion_Count'].values[0]),
            LABELS_KEYS.RL_CAVS:int(s_w_df['PD_L_LOB_RL_Cavitation'].values[0]), 
            LABELS_KEYS.RL_CONS:int(s_w_df['PD_L_LOB_RL_Consolidation'].values[0]), 
            LABELS_KEYS.RL_CONG:int(s_w_df['PD_L_LOB_RL_Conglomeration'].values[0]),
            LABELS_KEYS.RL_TBDS:int(s_w_df['PD_L_LOB_RL_Tree_in_Bud'].values[0]),
            
            LABELS_KEYS.LU_NODS:int(s_w_df['PD_L_LOB_LU_Lesion_Count'].values[0]),
            LABELS_KEYS.LU_CAVS:int(s_w_df['PD_L_LOB_LU_Cavitation'].values[0]), 
            LABELS_KEYS.LU_CONS:int(s_w_df['PD_L_LOB_LU_Consolidation'].values[0]), 
            LABELS_KEYS.LU_CONG:int(s_w_df['PD_L_LOB_LU_Conglomeration'].values[0]),
            LABELS_KEYS.LU_TBDS:int(s_w_df['PD_L_LOB_LU_Tree_in_Bud'].values[0]),
            
            LABELS_KEYS.LL_NODS:int(s_w_df['PD_L_LOB_LL_Lesion_Count'].values[0]),
            LABELS_KEYS.LL_CAVS:int(s_w_df['PD_L_LOB_LL_Cavitation'].values[0]), 
            LABELS_KEYS.LL_CONS:int(s_w_df['PD_L_LOB_LL_Consolidation'].values[0]), 
            LABELS_KEYS.LL_CONG:int(s_w_df['PD_L_LOB_LL_Conglomeration'].values[0]),
            LABELS_KEYS.LL_TBDS:int(s_w_df['PD_L_LOB_LL_Tree_in_Bud'].values[0])}


def data_augmenting(image):
    noise_sig = np.random.rand()*0.025
    flip_ax = np.random.randint(0,3)    
    print('offset_sig',offset_sig,' noise_sig', noise_sig,'flip_ax', flip_ax)
    dice_toss_transforms =  np.random.rand(3) 
    img = add_gaussian_noise(image,sigma = noise_sig) if dice_toss_transforms[0] > 0.5 else image
    img = add_gaussian_offset(image, sigma =  offset_sig ) if dice_toss_transforms[1] > 0.5 else img
    img = flip(img, axis = flip_ax) if dice_toss_transforms[2] > 0.5 else img

    return img


def read_fn(file_references, mode, params=None):
    """
    file_references: file_path img for the moment
    """
    masks_dir = params['masks_dir']
    for img_path in file_references:
        #print('Read fn img_path ',img_path )
        img, mask = read_mask_from_image_as_np(img_path, masks_dir)
        #TODO Should cast in preprocessing to apply HP for GPU
        img = normalise_one_one(img) #Following the literature this kind of normalization should work 
        
        imag_m= sitk.GetImageFromArray(mask.transpose())
        imag = sitk.GetImageFromArray(img.transpose())
        
        sitk.WriteImage(imag_m, os.path.join(ORIGINAL_DATA_PATH,'TRASH_IMAGES',img_path.split(os.sep)[-1] ))
        sitk.WriteImage(imag, os.path.join(ORIGINAL_DATA_PATH,'TRASH_IMAGES','r_'+img_path.split(os.sep)[-1] ))
        
        
        if mode == tf.estimator.ModeKeys.TRAIN:
          offset_sig = np.random.rand()*0.1
          noise_sig = np.random.rand()*0.075
          flip_ax = np.random.randint(0,3)    
          
          #img = add_gaussian_offset(img, sigma =  offset_sig )
          img = add_gaussian_noise(img,sigma = noise_sig)
          img,mask = flip([img,mask], axis = flip_ax)

        
        img = img[:,:,:,np.newaxis]
        mask = mask[:,:,:,np.newaxis]

        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': img}}
    
        # Labels:
        y = (mask > 0).astype(np.int32)
        imag_m= sitk.GetImageFromArray(y[:,:,:,0].transpose())
        imag = sitk.GetImageFromArray(img[:,:,:,0].transpose())
        
        sitk.WriteImage(imag_m, os.path.join(ORIGINAL_DATA_PATH,'TRASH_IMAGES','a_'+img_path.split(os.sep)[-1] ))
        sitk.WriteImage(imag, os.path.join(ORIGINAL_DATA_PATH,'TRASH_IMAGES','ra_'+img_path.split(os.sep)[-1] ))
        
        # If training should be done on image patches for improved mixing, 
        # memory limitations or class balancing, call a patch extractor
        if params['extract_examples']:
            images = extract_random_example_array(
                img,
                example_size=params['example_size'],
                n_examples=params['n_examples'])
            
            # Loop the extracted image patches and yield
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.int32)}}
                     
        else:
            yield {'features': {'x': img.astype(np.float16) },
                   'labels': {'y': y.astype(np.int32), 'Name': img_path.split(os.sep)[-1] }}

    return

def decode(file_ref):
    print('FILE_REFERENCE 22',type(file_ref) )
    #file_r = file_ref.decode("utf-8")
    print('FILE_REF',file_ref)
    img = read_itk_img_for_vnet(file_ref)
    return img
    


def read_itk_img_decode_for_vnet(img_path, mode,from_tensor = True):
    """
    Till now can handle any medical format sotred locally and .mhd files stored in the GCP
    """ 
    #print('IMG_PATH_DECODER', img_path)   
    #print('IMG_PATH_DECODER_TYPE', type(img_path) ) 
    img = sitk.ReadImage(img_path.decode("utf-8") if from_tensor else img_path ) 
    img_arr = sitk.GetArrayFromImage(img)
    img_arr = img_arr.transpose()
    img_arr = normalise_one_one(img_arr)
    print('MODE', mode,tf.estimator.ModeKeys.TRAIN,'PATH',img_path)
    if mode.decode() == tf.estimator.ModeKeys.TRAIN:
        #print('TO AUGMENT', img_path)
        img_arr = data_augmenting(img_arr)
    img_arr = img_arr[:,:,:,np.newaxis]
    return img_arr

def get_anotations_labels_decode(img_path,  annot_file = '../../TrainingSet_metaData_extra.csv', from_tensor = True):
    #annot_file = '../../TrainingSet_metaData_extra.csv'
    if isinstance(annot_file, str):
        with file_io.FileIO(annot_file, mode ='r') as f:
          annotatations_pandas = pd.read_csv(f)
    else:
        annotatations_pandas = annot_file
      

    file_reference = img_path.decode("utf-8") if from_tensor else img_path
    #print('FILE_REFERENCE', file_reference)
    img_fields = file_reference.split(os.sep)[-1]
    name = img_fields+'.gz'

    anotations = annotatations_pandas[annotatations_pandas['Filename']== name]
    annotation_severity = anotations[IMAGE_CLEF_KEYS.SEVERITY].values[0]
    annotation_left_lung = anotations[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED].values[0]
    annotation_right_lung = anotations[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED].values[0]
    annotation_lung_capacity = anotations[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE].values[0]
    anotation_calcification = anotations[IMAGE_CLEF_KEYS.CTR_CALCIFICATION].values[0]
    anotation_pleurisy = anotations[IMAGE_CLEF_KEYS.CTR_PLEURISY].values[0]
    anotation_caverns = anotations[IMAGE_CLEF_KEYS.CTR_CAVERNS].values[0]
    
    y = np.array( [annotation_severity, 
                   annotation_left_lung,
                   annotation_right_lung,
                   annotation_lung_capacity,
                   anotation_calcification,
                   anotation_pleurisy,
                   anotation_caverns] )
    print(name,'Y VALUE---------------------------------', y)
    return y,name


 
def read_fn_lbels_files(file_references, mode, params=None):  
    for file in file_references:
        yield file

def data_workaround(img_path_dummy, mode,annotation_file = '../../TrainingSet_metaData_extra.csv'):
    print('annotation_file',type(annotation_file))
    img = read_itk_img_decode_for_vnet(img_path_dummy, mode, from_tensor= True)
    y,name = get_anotations_labels_decode(img_path_dummy, from_tensor=True, annot_file= annotation_file)
    
    return img.astype(np.float32), y.astype(np.float32), str(name)



def data_fake(img_path_dummy, anotations_file):
    print('DATA FAKE')
    path_img = glob('../../TrainingSet_2mm/*.nii')
    path_img = random.choice(path_img)
    img = read_itk_img_decode_for_vnet(path_img, from_tensor=False)
    y, name = get_anotations_labels_decode(path_img, from_tensor = False, annot_file=anotations_file)
    
    return img, y, str(name)


def get_all(img_path, anntoations_file):
    y,name = get_anotations_labels_decode(img_path, from_tensor=False, annot_file=anntoations_file)
    return read_itk_img_decode_for_vnet(img_path, from_tensor=False), y, name


def read_fn_lbels_mp(file_references, mode, params=None):
    annot_file = params['reports_path']
    with Pool(5) as p:
        
        out = p.starmap(get_all, zip(file_references,[annot_file]*len(file_references) ))
        #multiple_results = [p.apply_async(os.getpid, ()) for i in range(4)]
        #print('yeah',[res.get(timeout=1) for res in multiple_results])

    print('OUT type', type(out))
    for path in range(0,len(out)):
        return {'features': {'x': out[path][0]},'labels': {'y': out[path][1].astype(np.float32), 'Name': out[path][2]}}
        
    return
        
    
    #for out in outs:
            #return {'features': {'x': out[0]},'labels': {'y': out[1].astype(np.float32), 'Name': out[2]}}
        

def read_fn_lbels_light(filename):
    img = tf.py_function(read_itk_img_decode_for_vnet, [filename], [tf.float32])
    y, name = tf.py_function(get_anotations_labels_decode, [filename], [tf.float32, tf.string])
    
    return  {'features': {'x': img},'labels': {'y': y, 'Name': name}}
    #return img,y.astype(np.float32),name



def input_function_labels(file_references, mode, params=None): 
    with tf.compat.v1.name_scope('input'):
        print('INDICES',tf.constant(file_references, dtype = tf.string) )
        
        dataset = tf.data.Dataset.list_files('../../TrainingSet_2mm/*.nii')
        dataset = dataset.map(read_fn_lbels_light )
        return dataset      
        

def read_fn_lbels(file_references, mode, params=None):
    """
    file_references: file_path img for the moment
    """
    #with tf.device('/cpu:0'):
    file_references = file_references if isinstance(file_references, list) else [file_references]
    annot_file = params['reports_path']
    with file_io.FileIO(annot_file, mode ='r') as f:
      #data = pd.read_csv(f)
      annotatations_pandas = pd.read_csv(f)

    for img_path in file_references:
        img = read_itk_img_for_vnet(img_path) if img_path.endswith(".nii") else dec_serial(img_path).T
        img = normalise_one_one(img)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            img = data_augmenting(img)

        img = img[:,:,:,np.newaxis]
        
        # If in PREDICT mode, yield the image (because there will be no label
        # present). Additionally, yield the sitk.Image pointer (including all
        # the header information) and some metadata (e.g. the subject id),
        # to facilitate post-processing (e.g. reslicing) and saving.
        # This can be useful when you want to use the same read function as 
        # python generator for deployment.
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': img}}
    
        # Labels:
        img_fields = img_path.split(os.sep)[-1]
        name = img_fields+'.gz'
    
    
        anotations = annotatations_pandas[annotatations_pandas['Filename']== name]
        annotation_severity = anotations[IMAGE_CLEF_KEYS.SEVERITY].values[0]
        annotation_left_lung = anotations[IMAGE_CLEF_KEYS.CTR_LEFT_LUNG_AFFECTED].values[0]
        annotation_right_lung = anotations[IMAGE_CLEF_KEYS.CTR_RIGHT_LUNG_AFFECTED].values[0]
        annotation_lung_capacity = anotations[IMAGE_CLEF_KEYS.CTR_LUNG_CAPACITY_DECREASE].values[0]
        anotation_calcification = anotations[IMAGE_CLEF_KEYS.CTR_CALCIFICATION].values[0]
        anotation_pleurisy = anotations[IMAGE_CLEF_KEYS.CTR_PLEURISY].values[0]
        anotation_caverns = anotations[IMAGE_CLEF_KEYS.CTR_CAVERNS].values[0]
        
        y = np.array( [annotation_severity, 
                       annotation_left_lung,
                       annotation_right_lung,
                       annotation_lung_capacity,
                       anotation_calcification,
                       anotation_pleurisy,
                       anotation_caverns] )#[np.newaxis] #TODO axis
        print(name,'Y VALUE---------------------------------', y)
#            
#  
        if params['extract_examples']:
            images = extract_random_example_array(
                img,
                example_size=params['example_size'],
                n_examples=params['n_examples'])
            
            # Loop the extracted image patches and yield
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.float32)}}
                     
        # If desired (i.e. for evaluation, etc.), return the full images
        else:
            yield {'features': {'x': img},
                   'labels': {'y': y.astype(np.float32), 'Name': name}}

    return


def img_vol_to_tfrecord_2(img_paths, tfrecords_filename,mask_paths = None, writer = None, chunks = 5):
    """
    img_paths and mask_paths in pairs
    Follow same rules that model.read_and_decode
    """
    
    chunk_size = int(np.round(len(img_paths) / float(chunks) ))
    chunk_lists = [ img_paths[x:x+chunk_size] for x in range(0,len(img_paths),chunk_size) ]
    
    mask_paths = [None] * len(img_paths) if mask_paths is None else mask_paths
    masks_list = [ mask_paths[x:x+chunk_size] for x in range(0,len(mask_paths),chunk_size) ]
    

    #writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    
    for i,lst in enumerate(chunk_lists):
        wr_name = tfrecords_filename+str(i)+'.tfrecords'
        print(wr_name)
        writer = tf.io.TFRecordWriter(wr_name)
    
        for j, img_path in enumerate(lst):
            print(img_path)
            img_np, msk_np = read_and_decode(img_path, masks_list[i][j])
            assert (img_np.shape == msk_np.shape)
            
            name = img_path.split(os.sep)[-1][:-4]
            
            height, width, depth, channels = img_np.shape
            
            #img_raw = img_np.tostring()
            #msk_raw = msk_np.tostring()
        
            example = tf.train.Example(features=tf.train.Features(feature={
            'ID':_bytes_feature(name),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'channels':_int64_feature(channels),
            'image_raw': _float_feature(img_np.ravel()),
            'mask_raw': _float_feature(msk_np.ravel() )}))
        
            writer.write(example.SerializeToString())
    
        writer.close()

def img_vol_to_tfrecord(img_paths, tfrecords_filename,mask_paths = None, writer = None, chunks = 5):
    """
    img_paths and mask_paths in pairs
    Follow same rules that model.read_and_decode
    """
    chunk_size = int(np.round(len(img_paths) / float(chunks) ))
    chunk_lists = [ img_paths[x:x+chunk_size] for x in range(0,len(img_paths),chunk_size) ]
    
    mask_paths = [None] * len(img_paths) if mask_paths is None else mask_paths
    masks_list = [ mask_paths[x:x+chunk_size] for x in range(0,len(mask_paths),chunk_size) ]
    
    for i,lst in enumerate(chunk_lists):
        wr_name = tfrecords_filename+str(i)+'.tfrecords'
        print(wr_name)
        writer = tf.io.TFRecordWriter(wr_name)
        
        
        for j, img_path in  enumerate(lst):
            img_np, msk_np = read_and_decode(img_path, masks_list[i][j])
            assert (img_np.shape == msk_np.shape)
            
            name = img_path.split(os.sep)[-1][:-4]
            
            height, width, depth, channels = img_np.shape
            
            img_raw = img_np.tostring()
            msk_raw = msk_np.tostring()
        
            example = tf.train.Example(features=tf.train.Features(feature={
            'ID':_bytes_feature(name),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'channels':_int64_feature(channels),
            'image_raw': _bytes_feature(img_raw),
            'mask_raw': _bytes_feature(msk_raw )}))
        
            writer.write(example.SerializeToString())
    
        writer.close()
        
        
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value] ))



def img_vol_to_tfrecord_humans(img_paths, tfrecords_filename, writer = None, chunks = 5, report = '../../TrainingSet_metaData_extra.csv'):
    """
    img_paths and mask_paths in pairs
    Follow same rules that model.read_and_decode
    """
    chunk_size = int(np.round(len(img_paths) / float(chunks) ))
    chunk_lists = [ img_paths[x:x+chunk_size] for x in range(0,len(img_paths),chunk_size) ]
    
    for i,lst in enumerate(chunk_lists):
        wr_name = tfrecords_filename+str(i)+'.tfrecords'
        print(wr_name)
        writer = tf.io.TFRecordWriter(wr_name)
        
        for j, img_path in  enumerate(lst):
            labels = get_anotations_labels_decode(img_path, report, from_tensor = False )
            
            img_np = read_itk_img_for_vnet(img_path) #return the image tansposed
            img_np = img_np[:,:,:,np.newaxis]
            name = img_path.split(os.sep)[-1][:-4]
            print("NAME",name)
            
            height, width, depth, channels = img_np.shape            
            label = np.array(labels[0], dtype=img_np.dtype)

        
            example = tf.train.Example(features=tf.train.Features(feature={
            'ID':_bytes_feature(name.encode()),
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'depth': _int64_feature(depth),
            'channels':_int64_feature(channels),
            'labels':_float_feature(label),
            'image_raw': _float_feature(img_np.ravel()),
             }))
        
            writer.write(example.SerializeToString())
    
        writer.close()



def decode_2(serialized_example, shape=[180, 180, 165, 1]):
    # Decode examples stored in TFRecord
    # NOTE: make sure to specify the correct dimensions for the images
    features = tf.io.parse_single_example(
        serialized=serialized_example,
        features={'image_raw': tf.io.FixedLenFeature(shape, tf.float32),
                  'labels': tf.io.FixedLenFeature([7], tf.float32), 
                  'ID':tf.io.FixedLenFeature((), tf.string, "")})
    
    "Normalization and augmenting"
    

    # NOTE: No need to cast these features, as they are already `tf.float32` values.
    return { 'features':{'x':features['image_raw']}, 'labels': {'y': features['labels'], 'Name': features['ID'] } } 

def decode_N(serialized_example, return_name = False):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)
    
    name = (example.features.feature['ID'].bytes_list.value[0])
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0])
    depth = int(example.features.feature['depth'].int64_list.value[0])
    channels = int(example.features.feature['channels'].int64_list.value[0])
    
    img_string = (example.features.feature['image_raw'].bytes_list.value[0])
    annotation_string = (example.features.feature['mask_raw'].bytes_list.value[0])
    
    img_1d = np.fromstring(img_string, dtype=np.int16)
    reconstructed_img = img_1d.reshape((height, width, depth,channels))
    
    annotation_1d = np.fromstring(annotation_string, dtype=np.uint8)
    reconstructed_annotation = annotation_1d.reshape((height, width, depth,channels))
    
    if return_name:
        return reconstructed_img, reconstructed_annotation, name
    return reconstructed_img, reconstructed_annotation 
    
   
def tfrecord_to_image_itk_image(tfrecords_file, imgs_to_return = 100):
    record_iterator = tf.compat.v1.python_io.tf_record_iterator(path=tfrecords_file)

    reconstructed_images = []
    
    while len(reconstructed_images) < imgs_to_return:
        string_record = next(record_iterator)
        reconstructed_img, reconstructed_annotation, name = decode_N(string_record,return_name=True)
        
        reconstructed_images.append({'ID':name,
                                     'img':sitk.GetImageFromArray(reconstructed_img[:,:,:,0].transpose()),
                                     'mask':sitk.GetImageFromArray(reconstructed_annotation[:,:,:,0].transpose())})
    
    return reconstructed_images
    


# Generator function
def f():
    fn = read_fn_lbels(file_references=glob('iTrain_images/*img.mhd'),
                 mode=tf.estimator.ModeKeys.TRAIN, 
                 params=None)
    
    ex = next(fn)
    # Yield the next image
    yield ex



def resample_image(itk_image, out_size = None, out_spacing=(1.0, 1.0, 1.0), is_label=False, deafault_value = None):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))] if out_size is None else out_size

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    default_value = itk_image.GetPixelIDValue() if deafault_value is None else deafault_value
    resample.SetDefaultPixelValue(default_value)
    
    

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
    
    resampled = resample.Execute(itk_image)

    return sitk.Cast(resampled, itk_image.GetPixelID())
  

def reslice_image(itk_image, itk_ref, is_label=False, deafault_value = None):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(itk_ref)
    default_value = itk_image.GetPixelIDValue() if deafault_value is None else deafault_value
    resample.SetDefaultPixelValue(default_value) 

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)
        
    resampled = resample.Execute(itk_image)
    
    sitk.WriteImage(resampled, '/tmp/'+str(itk_image.GetPixelIDValue())+'resampled.mhd')
    return sitk.Cast(resampled, itk_image.GetPixelID())


def f(x):
    return x*x


# class IteratorInitializerHook(tf.train.SessionRunHook):
class IteratorInitializerHook(tf.estimator.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

def entra_test():
    iterator_initializer_hook = IteratorInitializerHook()
    
    def train_inputs():
        dataset = tf.data.TFRecordDataset(glob('/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_2mm_serial/*.tfrecords')).map(decode_2)
        dataset = dataset.repeat(None)
        dataset = dataset.batch(8)
        dataset = dataset.prefetch(1)
    
        iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        next_dict = iterator.get_next()
    
        # Set runhook to initialize iterator
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(iterator.initializer)
    
        # Return batched (features, labels)
        return next_dict['features'], next_dict.get('labels')

    # Return function and hook
    return train_inputs, iterator_initializer_hook


def resample_dataset(reference_image_path, list_of_images, output_dir, resample_to = [1.,1.,1.], deafault_value_reference = -3024,
                     deafault_value_resampled = -3024, out_extension = '.nii'):
    """
    deafault_value_resampled = 0 if  'Mask' in output_dir else -3024
    resmaple_to: In mm
    """
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
     
    ref_image = sitk.ReadImage(reference_image_path)
    ref_image_resampled = resample_image(ref_image, out_spacing=resample_to, deafault_value = deafault_value_reference )
    sitk.WriteImage(ref_image, '/tmp/ref.nii')
    pixel_type_output = sitk.sitkInt16 if deafault_value_resampled == -3024 else sitk.sitkUInt8
  
    label = False if deafault_value_resampled == -3024 else True
    
    
    for image_path in list_of_images:
      name = image_path.split(os.sep)[-1].split('.')[0]+out_extension
      image = sitk.ReadImage(image_path)
      image = sitk.Cast(image, pixel_type_output)
      #resliced_image = reslice_image(image, ref_image_resampled, is_label=label, deafault_value = deafault_value_resampled)
      resliced_image = resample_image(image, out_size = ref_image_resampled.GetSize(),out_spacing=resample_to, is_label=label, deafault_value = deafault_value_resampled  )
      print(str(resample_to )+'mm',name+': {} {}'.format( resliced_image.GetSize(),  resliced_image.GetSpacing()))
      sitk.WriteImage(resliced_image, os.path.join(output_dir, name))
      

def center_images(fixed_image, moving_image, label=False, defaul_value=None):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)
    
    if label:
        interpolation = sitk.sitkNearestNeighbor
    else:
        interpolation = sitk.sitkBSpline        

    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, interpolation,
                                     defaul_value, moving_image.GetPixelID())
    
    return moving_resampled


def from_tensor_to_image(tensor_image, name):
    print("TENSOR",tensor_image)
    print("NAME_Tensor", name)
    image = sitk.GetImageFromArray(tensor_image.transpose())
    print("NAME_Tensor", name)
    sitk.WriteImage(image, name)
    print("NAME_Tensor", name)
    return 'yes!'


def get_itk_image(np_image, name):
    np_image = np_image[0, :, :, :, 0].transpose()
    itk_image = sitk.GetImageFromArray(np_image)
    sitk.WriteImage(itk_image, name)


def adamW(learning_rate, weight_decay, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False,
          name='AdamW'):  # parameterss order are changed
    return tf.contrib.opt.AdamWOptimizer(weight_decay, learning_rate=learning_rate,
                                         beta1=beta1, beta2=beta2, epsilon=epsilon,
                                         use_locking=use_locking, name=name)

if __name__ == '__main__':

  extension = 'nii.gz'
#  
  images_trn_1 = glob(os.path.join(ORIGINAL_DATA_PATH, 'TrainingSet_1_of_2/')+'*'+extension)
  images_trn_2 = glob(os.path.join(ORIGINAL_DATA_PATH, 'TrainingSet_2_of_2/')+'*'+extension)
  images_trn = images_trn_1 + images_trn_2
  masks_trn = glob(os.path.join(ORIGINAL_DATA_PATH, 'TrainingSet_Modified_Masks/')+'*'+extension)

  for img in images_trn:
      name = img.split(os.sep)[-1].split('.')[0]
      mask = os.path.join(ORIGINAL_DATA_PATH, 'TrainingSet_Modified_Masks/'+ name+"_1.nii")
      mask = sitk.ReadImage(mask) > 0
      img = sitk.ReadImage(img)
      mask.CopyInformation(img)
      mask = sitk.Cast(mask, sitk.sitkUInt8)
      img,mask = crop_images_to_lung_mask(img, mask)
      resampled = resample_sitk_image_size(img, size=(128,128,128), interpolator='bspline')
      resampled_mask = resample_sitk_image_size(mask, size=(128,128,128), interpolator='nearest')
      sitk.WriteImage(resampled, "/home/pedro/Documents/projects/bodyct-tuberculosis-multitask/TrainingSet_ISO_NS/"+name+".nii")
      sitk.WriteImage(resampled_mask, '//home/pedro/Documents/projects/bodyct-tuberculosis-multitask/TrainingSet_MASKS_ISO_NS/'+name+'.nii')

