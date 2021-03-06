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
# path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
# os.chdir(path)


path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'
os.chdir(path)

X=np.load('permutedseqMNIST_alldigits.npy')
# X=np.load('mnist_trakr_X_alldigits.npy')
# X=standardize_data(X)
y=np.load('mnist_trakr_labels_alldigits.npy')

#%% performance and evaluation - metrics
accuracy,aucvec=cross_val_metrics_naiveB(X,y,n_classes=10,splits=10)


#%%
performance_metrics=dict()
performance_metrics['accuracy']=accuracy
performance_metrics['auc']=aucvec


#%%

import pickle

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_nb_permutedmnist_noiselimitupto_5', 'wb') as f:
    pickle.dump(metrics, f)


#%% Noisy Inputs


path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'

os.chdir(path)
y=np.load('mnist_trakr_labels_alldigits.npy')
level=np.linspace(1,6,50)
metrics=dict()

for loop in range(len(level)):
    X=np.load('permutedseqMNIST_alldigits.npy')
    sigma=level[loop]
    X=add_noise(X,sigma) 
    accuracy,aucvec=cross_val_metrics_naiveB(X,y,n_classes=10,splits=10)
    metrics[f'Noiselevel {level[loop]} - accuracy']=accuracy
    metrics[f'Noiselevel {level[loop]} - auc']=aucvec
    print(f'On Noiselevel {level[loop]}')


























