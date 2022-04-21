#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:54:57 2021

@author: furqanafzal
"""
#%% importing modules
import os
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr'
os.chdir(path)
import numpy as np
import matplotlib.pylab as plt
import modules
import importlib
importlib.reload(modules) 
from modules import dynamics,add_noise,train_test_loop,cross_val_metrics_trakr,standardize_data
from modules import permutedseqMNIST,train_test_loop_noisyinput
import pickle

#%% presaved MNIST train and test digits
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'
os.chdir(path)

x_train=np.load('permutedseqMNIST_alldigits.npy')
y_train=np.load('mnist_trakr_labels_alldigits.npy')

# x_train=standardize_data(x_train)

#%% add noise to training digits optionally
# sigma=1
# x_train=add_noise(x_train,sigma)

#%% trakr initializations and train & test loop

N=30 # number of neurons in the RNN
N_out=1
g=1.4 # gain
tau=1 # tau
delta = .3 # delta for Euler's method
alpha=1 # alpha for regularizer
totaltime=np.size(x_train,1)
iterations=1
learning_error_matrix=train_test_loop(iterations,x_train,N,N_out,g,tau,delta,alpha,totaltime)
np.save('learningerror_permutedseqmnist.npy',learning_error_matrix)

#%% load presaved learning error matrix
# path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
# os.chdir(path)

# x_train=np.load('mnist_trakr_learningerror_alldigits.npy')
# y_train=np.load('mnist_trakr_labels_alldigits.npy')


#%% classification and evaluation - metrics
accuracy,aucvec=cross_val_metrics_trakr(x_train,y_train,n_classes=10, splits=10)


#%%
# performance_metrics={'accuracy-svm':accuracy,'auc-svm':aucvec}

# performance_metrics=dict()
performance_metrics['accuracy-knn']=accuracy
performance_metrics['auc-knn']=aucvec

#%%

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_trakr_pseqmnist_noiselimitupto_2', 'wb') as f:
    pickle.dump(metrics, f)

#%%

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/metrics_trakr_mnist', 'rb') as f:
    loaded_dict = pickle.load(f)


#%% get permuted sequential MNIST from seq MNIST

# data=permutedseqMNIST(x_train)

 #%%
# c=1
# for i in range(51,55):

#     plt.subplot(5,1,c)
#     plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
#     plt.axis("off")
#     plt.title(f"Permuted sequence of the digit '{y_train[i]}' ")
#     plt.show()
#     c+=1
    
##################################################################################

################################################################
# Noisy Inputs
################################################################

#%%
level=np.linspace(0,5,50)
# level=[2]
totaltime=784
N=30 # number of neurons in the RNN
N_out=1
g=1.4 # gain
tau=1 # tau
delta = .3 # delta for Euler's method
alpha=1 # alpha for regularizer
totaltime=np.size(x_train,1)
iterations=1
metrics=dict()

for loop in range(len(level)):
    x_train=np.load('permutedseqMNIST_alldigits.npy')
    sigma=level[loop]
    x_train=add_noise(x_train,sigma)
    learning_error_matrix=train_test_loop_noisyinput(iterations,x_train,N,N_out,g,tau,delta,alpha,totaltime)
    accuracy,aucvec=cross_val_metrics_trakr(learning_error_matrix,y_train,n_classes=10, splits=10)
    metrics[f'Noiselevel {level[loop]} - accuracy']=accuracy
    metrics[f'Noiselevel {level[loop]} - auc']=aucvec
    print(f'On Noiselevel {level[loop]}')

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_trakr_pseqmnist_noiselimitupto_5', 'wb') as f:
    pickle.dump(metrics, f)

#%%

















