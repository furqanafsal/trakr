#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:49:33 2021

@author: furqanafzal
"""
#%%modules
import os
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr'
os.chdir(path)
import numpy as np
import matplotlib.pylab as plt
import modules
import importlib
importlib.reload(modules) 
from modules import add_noise,standardize_data,cross_val_metrics
import pyinform as pyinf


#%% load data
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
os.chdir(path)

X=np.load('mnist_trakr_X_alldigits.npy')
X=standardize_data(X)
y=np.load('mnist_trakr_labels_alldigits.npy')

#%% performance and evaluation - metrics
accuracymat=[] 
aucmat=[]
for k in range(np.size(X,0)):
    distmat=[]
    for i in range(np.size(X,0)):
        distmat.append(pyinf.mutual_info(X[i,:], X[k,:]))
        print(i)
    distmat=np.array(distmat).reshape(-1,1)
    accuracy,aucvec=cross_val_metrics(distmat,y,n_classes=10)
    accuracymat.append(accuracy),aucmat.append(aucvec)