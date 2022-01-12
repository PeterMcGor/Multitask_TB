#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:54:57 2019

@author: pedro
"""

import numpy as np
import pandas as pd
import time

from collections  import Counter
from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection.measures import get_combination_wise_output_matrix



def iterative_train_test_split(X, y, test_size, rs = 42):
    """Iteratively stratified train/test split
    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set
    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
    """

    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[test_size, 1.0-test_size], random_state=rs)
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_test, y_test = X[test_indexes, :], y[test_indexes, :]

    return X_train, y_train, X_test, y_test



anot = pd.read_csv('/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/TrainingSet_metaData_extra.csv')

X = np.array([anot.Filename.values])
X = X.T

y = np.array([anot.SVR_Severity.values, anot.CTR_LeftLungAffected.values, 
              anot.CTR_RightLungAffected.values, anot.CTR_LungCapacityDecrease.values, 
              anot.CTR_Calcification.values, anot.CTR_Pleurisy.values, anot.CTR_Pleurisy.values])

y = y.T
y[:,0] = y[:,0] == 'LOW'
y = y.astype(np.uint8)

X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size = 0.2)

pd.DataFrame(X_test).to_csv('/mnt/synology/bodyct/experiments/tuberculosis-chestct-t8411/imageclef2019/Val_Set_'+time.strftime("%Y%m%d%H%M%S")+'.csv')

d = pd.DataFrame({
    'train': Counter(str(combination) for row in get_combination_wise_output_matrix(y_train, order=2) for combination in row),
    'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for combination in row)}).T.fillna(0.0)
    