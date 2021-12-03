#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:15:32 2021

@author: furqanafzal
"""
#%%modules
import numpy as np
import matplotlib.pyplot as plt
import os 
os.chdir('/Users/furqanafzal/Documents/furqan/MountSinai/Research/Code/trakr')
import numpy as np
import matplotlib.pylab as plt
from IPython import get_ipython
from sklearn.metrics import mean_squared_error
from scipy import signal
import pandas as pd
from scipy.signal import sosfiltfilt
from scipy.signal import savgol_filter
from modules import cross_val_metrics_naiveB
from sklearn import preprocessing
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
from modules import cross_val_metrics
import pyinform as pyinf

path='/Users/furqanafzal/Documents/furqan/MountSinai/Research/ComputationalNeuro/erin_collab/variabledata'
os.chdir(path)

#%% load data
X=np.load('mnist_trakr_X_alldigits.npy')
X=stats.zscore(X,axis=1)
y=np.load('mnist_trakr_labels_alldigits.npy')

#%% performance and evaluation - metrics
accuracy,aucvec=cross_val_metrics_naiveB(X,y,n_classes=10)































