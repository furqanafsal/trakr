#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 15:22:47 2021

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


#%% load data
# path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'

os.chdir(path)

X=np.load('permutedseqMNIST_alldigits.npy')
# X=standardize_data(X)
y=np.load('mnist_trakr_labels_alldigits.npy')

#%% performance and evaluation - metrics
accuracymat=[] 
aucmat=[]
for k in range(np.size(X,0)):
    distmat=[]
    print(f'On Iteration {k}')
    for i in range(np.size(X,0)):
        distmat.append(np.linalg.norm(X[i,:] - X[k,:]))
    distmat=np.array(distmat).reshape(-1,1)  
    accuracy,aucvec=cross_val_metrics(distmat,y,n_classes=10,splits=10)
    accuracymat.append(accuracy),aucmat.append(aucvec)

#%%
print(np.mean(accuracymat))
print(np.mean(aucmat))
#%%
# performance_metrics=dict()
performance_metrics['accuracy-knn']=accuracymat
performance_metrics['auc-knn']=aucmat

#%%

import pickle

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_euc_permutedmnist_noiselimitupto_5', 'wb') as f:
    pickle.dump(metrics, f)
    
    
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
            distmat.append(np.linalg.norm(X[i,:] - X[k,:]))
        distmat=np.array(distmat).reshape(-1,1)  
        accuracy,aucvec=cross_val_metrics(distmat,y,n_classes=10,splits=10)
        accuracymat.append(accuracy),aucmat.append(aucvec)
    metrics[f'Noiselevel {level[loop]} - accuracy']=accuracymat
    metrics[f'Noiselevel {level[loop]} - auc']=aucmat
    print(f'On Noiselevel {level[loop]}')
    
    
    
    
    
    
    
    
    
