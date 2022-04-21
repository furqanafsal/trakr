#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 11:05:11 2021

@author: furqanafzal
"""
#%%modules
import _ucrdtw
import os
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr'
os.chdir(path)
import numpy as np
# import matplotlib.pylab as plt
import modules
import importlib
importlib.reload(modules) 
from modules import add_noise,standardize_data,cross_val_metrics

#%% load data
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'
os.chdir(path)

X=np.load('permutedseqMNIST_alldigits.npy')
# X=standardize_data(X)
y=np.load('mnist_trakr_labels_alldigits.npy')


#%% performance and evaluation - metrics

# meanauc for window 5 is 0.5829566666666666
# svm
# meanauc for window 10 is 0.5866572222222222
# svm
# meanauc for window 20 is 0.5869033333333332
# svm
# meanauc for window 50 is 0.5846750000000001
# svm
# meanauc for window 100 is 0.5775111111111111
# svm

accuracymat=[]
aucmat=[]
for k in range(np.size(X,0)):
    distmat=[]
    print(f'On iteration {k}')
    for i in range(np.size(X,0)):
        loc, dist = _ucrdtw.ucrdtw(X[i,:], X[k,:] ,20, False)
        distmat.append(dist)
        # print(i)
    distmat=np.array(distmat).reshape(-1,1)
    accuracy,aucvec=cross_val_metrics(distmat,y,n_classes=10,splits=10)
    accuracymat.append(accuracy),aucmat.append(aucvec)

#%%
# performance_metrics={'accuracy-svm':accuracy,'auc-svm':aucvec}

# performance_metrics=dict()
performance_metrics['accuracy-knn']=accuracymat
performance_metrics['auc-knn']=aucmat

#%%
import pickle

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_dtw_permutedmnist_noiselimitupto_5', 'wb') as f:
    pickle.dump(metrics, f)

#%%

# with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/metrics_trakr_mnist', 'rb') as f:
#     loaded_dict = pickle.load(f)

#%%

################################################################
# Noisy Inputs
################################################################

#%%

path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'

os.chdir(path)

X=np.load('permutedseqMNIST_alldigits.npy')
y=np.load('mnist_trakr_labels_alldigits.npy')
level=np.linspace(0,5,50)
metrics=dict()
# digits=[0,100,200,300,400,500,600,700,800,900]
digits=[0,500,900]

for loop in range(len(level)):
    accuracymat=[] 
    aucmat=[]
    X=np.load('permutedseqMNIST_alldigits.npy')
    sigma=level[loop]
    X=add_noise(X,sigma)
    for k in digits:
        distmat=[]
        for i in range(np.size(X,0)):
            loc, dist = _ucrdtw.ucrdtw(X[i,:], X[k,:] ,20, False)
            distmat.append(dist)
        distmat=np.array(distmat).reshape(-1,1)  
        accuracy,aucvec=cross_val_metrics(distmat,y,n_classes=10,splits=10)
        accuracymat.append(accuracy),aucmat.append(aucvec)
    metrics[f'Noiselevel {level[loop]} - accuracy']=accuracymat
    metrics[f'Noiselevel {level[loop]} - auc']=aucmat
    print(f'On Noiselevel {level[loop]}')
    














