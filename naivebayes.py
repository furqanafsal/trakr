#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:15:32 2021

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
from modules import add_noise,standardize_data,cross_val_metrics_naiveB


#%% load data
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
os.chdir(path)

X=np.load('mnist_trakr_X_alldigits.npy')
X=standardize_data(X)
y=np.load('mnist_trakr_labels_alldigits.npy')

#%% performance and evaluation - metrics
accuracy,aucvec=cross_val_metrics_naiveB(X,y,n_classes=10,splits=10)


#%%
performance_metrics=dict()
performance_metrics['accuracy']=accuracy
performance_metrics['auc']=aucvec


#%%

import pickle

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/metrics_nb_mnist', 'wb') as f:
    pickle.dump(performance_metrics, f)



























