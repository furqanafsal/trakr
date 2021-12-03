#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:53:57 2021

@author: furqanafzal
"""

#################################
# Based on the code provided by Fawaz et al related to their 2019 paper.
# https://link.springer.com/article/10.1007/s10618-019-00619-1
# https://github.com/hfawaz/dl-4-tsc
#################################

import os 
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
from scipy.stats import zscore
import matplotlib.pyplot as plt 
from sklearn import preprocessing
from modules import create_classifier,generateMNISTdata

#%%

path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
os.chdir(path)
x_train=np.load('mnist_trakr_X_alldigits.npy')
y_train=np.load('mnist_trakr_labels_alldigits.npy')
x_train=zscore(x_train,axis=1)
x_test,y_test=generateMNISTdata()
x_test=zscore(x_test,axis=1)

#%%
os.chdir('/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr')
n_classes = 10
## transform the labels from integers to one hot vectors
enc = preprocessing.OneHotEncoder(categories='auto')
enc.fit(y_train.reshape(-1,1))
y_train=enc.transform(y_train.reshape(-1, 1)).toarray()
y_test=enc.transform(y_test.reshape(-1, 1)).toarray()

# save orignal y because later we will use binary
y_true = np.argmax(y_test, axis=1)

#%%
if len(x_train.shape) == 2:  # if univariate
# add a dimension to make it multivariate with one dimension 
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

input_shape = x_train.shape[1:]

classifier = create_classifier('twiesn', input_shape, n_classes, path)

#%%
accuracy, aucvec=classifier.fit(x_train, y_train, x_test, y_test, y_true)
