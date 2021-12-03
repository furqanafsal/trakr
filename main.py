#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:54:57 2021

@author: furqanafzal
"""
#%% importing modules
import os
os.chdir('/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr')
import numpy as np
import matplotlib.pylab as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')
from sklearn.metrics import mean_squared_error
import mat73
from scipy.io import loadmat
from scipy import signal
import pandas as pd
from scipy.signal import sosfiltfilt
from scipy.signal import savgol_filter
from modules import dynamics,add_noise,train_test_loop,cross_val_metrics_trakr
from sklearn import preprocessing
import tensorflow as tf
from scipy import stats
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


#%% Manipulate working directories
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
os.chdir(path)

#%% load new MNIST digits

(X_arr, y_arr), (
    Xtest,
    ytest,
) = tf.keras.datasets.mnist.load_data()

X=np.zeros((10,100,28,28))
y=np.zeros((10,100))
for i in range(10):
      tempx = X_arr[np.where((y_arr == i ))]
      tempy= y_arr[np.where((y_arr == i ))]
      ind = np.random.choice(np.size(tempx,0), size=100, replace=False)
      X[i,:,:,:]=tempx[ind,:]
      y[i,:]=tempy[ind]

x_train=X.reshape(1000,784)
y_train=y.reshape(1000)

#%% presaved MNIST train and test digits

x_train=np.load('mnist_trakr_X_alldigits.npy')
y_train=np.load('mnist_trakr_labels_alldigits.npy')
x_test=np.load('mnist_trakr_Xtest_alldigits.npy')
y_test=np.load('mnist_trakr_ytest_alldigits.npy')
x_test=stats.zscore(x_test,axis=1)
x_train=stats.zscore(x_train,axis=1)

#%% add noise to training digits optionally
# sigma=1
# x_train=add_noise(x_train,sigma)

#%% trakr initializations and train & test loop

N=30 # number of neurons in the RNN
N_out=1
g=1.2 # gain
tau=1 # tau
delta = .3 # delta for Euler's method
alpha=1 # alpha for regularizer
totaltime=np.size(x_train,1)
learning_error_matrix=train_test_loop(x_train,N,N_out,g,tau,delta,alpha,totaltime)
np.save('learningerror_data.npy',learning_error_matrix) # save the learning error

#%% load presaved learning error matrix

x_train=np.load('mnist_trakr_learningerror_alldigits.npy')
y_train=np.load('mnist_trakr_labels_alldigits.npy')


#%% classification and evaluation - metrics
accuracy,aucvec=cross_val_metrics_trakr(x_train,y_train,n_classes=10)






















