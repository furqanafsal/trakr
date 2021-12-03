#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 08:33:19 2021

@author: furqanafzal
"""

import numpy as np
import pdb
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
import tensorflow as tf
# import _ucrdtw

#%% activity/dynamics loop
def dynamics(N_out,N,g,tau,delta,f,totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,freezew,t1_train,t2_train):
    for t in range(totaltime):
            r[:, t] = np.tanh(x).reshape(N,) # activation function to calculate rates
            z_out[:,t] = np.dot(w_out.T,r[:,t].reshape(N,)) # zi(t)=sum (Jij rj) over j
            x = x + (-x + np.dot(J,r[:,t]).reshape(N,1) + np.dot(w_in,f[:,t]).reshape(N,1))*(delta/tau) # Euler update for activity x
            error[:,t] = z_out[:,t] - f[:,t] # z(t)-f(t)
            c=1/(1+ r[:,t].T@regP@r[:,t]) # learning rate
            regP = regP - c*(regP@r[:,t].reshape(N,1)@r[:,t].T.reshape(1,N)@regP) # calculating P(t)
            delta_w=c*error[:,t].reshape(N_out,1)*(regP@r[:,t]).T.reshape(1,N) # calculating deltaW for the readout unit
        #    indices = np.random.choice(np.arange(delta_w.size), replace=False,
        #                           size=int(delta_w.size * 1)) #setting random percent delta_ws to zero, for decreasing learning
        #    delta_w[indices]=0
            learning_error[:,t] = np.sum(abs(delta_w),axis=1) # calculating learning error
            if freezew==0:
                if t>=t1_train and t <= t2_train:
                    w_out = w_out - delta_w.T # output weights being plastic
    return error,learning_error,z_out,w_out,x,regP

#%% add gaussian noise
def add_noise(x,sigma):
    noise=np.random.normal(0,sigma,size=(np.shape(x)))
    x=x+noise
    return x

#%% train and test loop for trakr
def train_test_loop(x_train,N,N_out,g,tau,delta,alpha,totaltime):
    learning_error_tot=np.zeros((np.size(x_train,0),np.size(x_train,0),totaltime))
    for i in range(np.size(x_train,0)): 
        regP=alpha*np.identity(N) # regularizer
        J = g*np.random.randn(N,N)/np.sqrt(N) # connectivity matrix J
        r = np.zeros((N, totaltime)) # rate matrix - firing rates of neurons
        x = np.random.randn(N, 1) # activity matrix before activation function applied
        z_out = np.zeros((N_out,totaltime)) # z(t) for the output read out unit
        error = np.zeros((N_out, totaltime)) # error signal- z(t)-f(t)
        learning_error = np.zeros((N_out, totaltime)) # change in the learning error over time
        w_out = np.random.randn(N, N_out)/np.sqrt(N) # output weights for the read out unit
        w_in = np.random.randn(N, N_out) # input weights
        f=x_train[i,:].reshape(1,-1)
        # pdb.set_trace()
        error,learning_error,z_out,w_out,x,regP=dynamics(N_out,N,g,tau,delta,f,
                        totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                        freezew=0,t1_train=0,
                        t2_train=totaltime)
        for j in range(np.size(x_train,0)):
            f=x_train[j,:].reshape(1,-1)
            error,learning_error,z_out,w_out,_,_=dynamics(N_out,N,g,tau,delta,f,
                        totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                        freezew=1,t1_train=0,
                        t2_train=totaltime)
            learning_error_tot[i,j,:]=learning_error
            print(j)
    return learning_error_tot

#%% cross validation metrics for trakr ; accuracy and auc
def cross_val_metrics_trakr(x,y,n_classes):
    skf = StratifiedKFold(n_splits=10)
    accuracy=[]
    aucvec=[]
    fpr = dict()
    tpr = dict()
    for k in range(np.size(x,0)):
        for train_ix, test_ix in skf.split(x[k,:,:],y):
            roc_auc = np.zeros((n_classes))
            # split data
            X_train, X_test = x[k,train_ix, :], x[k,test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            # fit model
            model = svm.SVC()
            model.fit(X_train, y_train)
            # evaluate model
            y_pred=model.predict(X_test)
            accuracy.append(accuracy_score(y_test, y_pred))
            bin_ytrue = label_binarize(y_test, classes=np.arange(0,n_classes))
            bin_ypred = label_binarize(y_pred, classes=np.arange(0,n_classes))
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(bin_ytrue[:, i], bin_ypred[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            aucvec.append(roc_auc)
        print(k)
    return accuracy,aucvec

#%% cross validation metrics for all other methods ; accuracy and auc
def cross_val_metrics(x,y,n_classes):
    skf = StratifiedKFold(n_splits=10)
    accuracy=[]
    aucvec=[]
    fpr = dict()
    tpr = dict()
    for train_ix, test_ix in skf.split(x,y):
        roc_auc = np.zeros((n_classes))
        # split data
        X_train, X_test = x[train_ix, :], x[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # fit model
        model = svm.SVC()
        model.fit(X_train, y_train)
        # evaluate model
        y_pred=model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        bin_ytrue = label_binarize(y_test, classes=np.arange(0,n_classes))
        bin_ypred = label_binarize(y_pred, classes=np.arange(0,n_classes))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(bin_ytrue[:, i], bin_ypred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        aucvec.append(roc_auc)
    return accuracy,aucvec



#%% cross validation metrics for Naive Bayes
def cross_val_metrics_naiveB(x,y,n_classes):
    skf = StratifiedKFold(n_splits=10)
    accuracy=[]
    aucvec=[]
    fpr = dict()
    tpr = dict()
    for train_ix, test_ix in skf.split(x,y):
        roc_auc = np.zeros((n_classes))
        # split data
        X_train, X_test = x[train_ix, :], x[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        # fit model
        model = GaussianNB()
        model.fit(X_train, y_train)
        # evaluate model
        y_pred=model.predict(X_test)
        accuracy.append(accuracy_score(y_test, y_pred))
        bin_ytrue = label_binarize(y_test, classes=np.arange(0,n_classes))
        bin_ypred = label_binarize(y_pred, classes=np.arange(0,n_classes))
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(bin_ytrue[:, i], bin_ypred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        aucvec.append(roc_auc)
    return accuracy,aucvec

#%% Generate MNIST data
def generateMNISTdata():
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

    x_test=X.reshape(1000,784)
    y_test=y.reshape(1000)
    return x_test,y_test


#%%

#################################
# Based on the code provided by Fawaz et al related to their 2019 paper.
# https://link.springer.com/article/10.1007/s10618-019-00619-1
# https://github.com/hfawaz/dl-4-tsc
#################################
import twiesn

def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    return twiesn.Classifier_TWIESN(output_directory, verbose=True)














        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    