#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:52:29 2021

@author: furqanafzal
"""
#################################
# Based on the code provided by Fawaz et al related to their 2019 paper.
# https://link.springer.com/article/10.1007/s10618-019-00619-1
# https://github.com/hfawaz/dl-4-tsc
#################################

#%% import modules
import os
path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr'
os.chdir(path)
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from modules import generateMNISTdata,standardize_data,add_noise
from sklearn.metrics import accuracy_score
import pickle
#%%
# path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
# os.chdir(path)
# x_train=np.load('mnist_trakr_X_alldigits.npy')
# y_train=np.load('mnist_trakr_labels_alldigits.npy')
# x_train=standardize_data(x_train)

#%%

path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'
os.chdir(path)

x_train=np.load('permutedseqMNIST_alldigits.npy')

# X=np.load('mnist_trakr_X_alldigits.npy')
# X=standardize_data(X)
y_train=np.load('mnist_trakr_labels_alldigits.npy')

#%% model init

input_shape=(np.size(x_train,1),1)
n_classes=10

## using an MLP

input_layer = keras.layers.Input(input_shape)

# flatten/reshape because when multivariate all should be on the same axis 
input_layer_flattened = keras.layers.Flatten()(input_layer)

layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

layer_2 = keras.layers.Dropout(0.2)(layer_1)
layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

layer_3 = keras.layers.Dropout(0.2)(layer_2)
layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

output_layer = keras.layers.Dropout(0.3)(layer_3)
output_layer = keras.layers.Dense(n_classes, activation='softmax')(output_layer)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

#%% model fitting and eval

skf = StratifiedKFold(n_splits=10,shuffle=True)
epochs=100
accuracy=[]
aucvec=[]
fpr = dict()
tpr = dict()

for train_idx, val_idx in skf.split(x_train, y_train):
    roc_auc = np.zeros((n_classes))
    batch_size = 32
#     print('\nFold ',j)
    X_train_cv = x_train[train_idx]
    y_train_cv = y_train[train_idx]
    X_valid_cv = x_train[val_idx]
    y_valid_cv= y_train[val_idx]
    callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),]
    model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],)
    history = model.fit(
    X_train_cv,
    y_train_cv,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data = (X_valid_cv, y_valid_cv),
    verbose=0)
    x_test,y_test=generateMNISTdata()
    y_pred=model.predict(x_test)
    y_bin = label_binarize(y_test, classes=np.arange(0,n_classes))
    accuracy.append(accuracy_score(np.argmax(y_bin,axis=1), np.argmax(y_pred,axis=1)))
    # print('iteration')
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    aucvec.append(roc_auc)

#%%

performance_metrics=dict()
performance_metrics['accuracy']=accuracy
performance_metrics['auc']=aucvec

#%%



with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_mlp_permutedmnist_noiselimitupto_2', 'wb') as f:
    pickle.dump(metrics, f)

################################################################################

################################################################
# Noisy Inputs
################################################################

#%%

path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results'
os.chdir(path)

x_train=np.load('permutedseqMNIST_alldigits.npy')
y_train=np.load('mnist_trakr_labels_alldigits.npy')


input_shape=(np.size(x_train,1),1)
n_classes=10

input_layer = keras.layers.Input(input_shape)

# flatten/reshape because when multivariate all should be on the same axis 
input_layer_flattened = keras.layers.Flatten()(input_layer)

layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

layer_2 = keras.layers.Dropout(0.2)(layer_1)
layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

layer_3 = keras.layers.Dropout(0.2)(layer_2)
layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

output_layer = keras.layers.Dropout(0.3)(layer_3)
output_layer = keras.layers.Dense(n_classes, activation='softmax')(output_layer)

model = keras.models.Model(inputs=input_layer, outputs=output_layer)

skf = StratifiedKFold(n_splits=10,shuffle=True)
epochs=100
level=np.linspace(0,5,50)
metrics=dict()

for loop in range(len(level)):
    accuracy=[]
    aucvec=[]
    fpr = dict()
    tpr = dict()
    sigma=level[loop]
    x_train=np.load('permutedseqMNIST_alldigits.npy')
    x_train=add_noise(x_train,sigma)
    for train_idx, val_idx in skf.split(x_train, y_train):
        roc_auc = np.zeros((n_classes))
        batch_size = 32
    #     print('\nFold ',j)
        X_train_cv = x_train[train_idx]
        y_train_cv = y_train[train_idx]
        X_valid_cv = x_train[val_idx]
        y_valid_cv= y_train[val_idx]
        callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=0),]
        model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],)
        history = model.fit(
        X_train_cv,
        y_train_cv,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_data = (X_valid_cv, y_valid_cv),
        verbose=0)
        x_test,y_test=generateMNISTdata()
        x_test=add_noise(x_test,sigma)
        y_pred=model.predict(x_test)
        y_bin = label_binarize(y_test, classes=np.arange(0,n_classes))
        accuracy.append(accuracy_score(np.argmax(y_bin,axis=1), np.argmax(y_pred,axis=1)))
        # print('iteration')
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        aucvec.append(roc_auc)
    metrics[f'Noiselevel {level[loop]} - accuracy']=accuracy
    metrics[f'Noiselevel {level[loop]} - auc']=aucvec
    print(f'On Noiselevel {level[loop]}')

with open('/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/trakr/neurips2022/data_results/noisyinput_metrics_mlp_permutedmnist_noiselimitupto_5', 'wb') as f:
    pickle.dump(metrics, f)






















