
"""
Created on Fri May 28 08:33:19 2021
##test
@author: furqanafzal
"""

import numpy as np
import matplotlib.pylab as plt
# import pdb
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
# import tensorflow as tf
from scipy import stats
from scipy import signal
from scipy.signal import sosfiltfilt
# import _ucrdtw

#%% activity/dynamics loop
def dynamics(i,N_out,N,g,tau,delta,inputs,targets,totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,freezew,t1_train,t2_train):
    print(f"Shape targets: {np.shape(targets)}")
    print(f"Shape w_in: {np.shape(w_in)}")
    plt.figure(figsize=(10,6))
    for t in range(totaltime):
        print(f"Fit time series | Step {t}/{totaltime}")
        r[:, t] = np.tanh(x).reshape(N,) # activation function to calculate rates
        z_out[:,t] = np.dot(w_out.T,r[:,t].reshape(N,)) # zi(t)=sum (Jij rj) over j
        x = x + (-x + np.dot(J,r[:,t]).reshape(N,1) + np.dot(w_in,inputs[:,t]).reshape(N,1))*(delta/tau) # Euler update for activity x
        error[:,t] = z_out[:,t] - targets[:,t] # z(t)-f(t)
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
    # #Update plot        
    # if t%200 == 0:
    plot_dynamics(i,totaltime,targets,z_out,learning_error)
            
    return error,learning_error,z_out,w_out,x,regP

#%%
def plot_dynamics(i,totaltime,targets,z_outmat,learning_error):

     linewidth=2; fontsize = 12
     time = np.linspace(0,totaltime/1000,totaltime).reshape(1,totaltime) # time vecto
     legend_entries = ['Signal','Label']

     plt.subplot(311)
     plt.plot(time[0,:],targets[0,:],color="red", linewidth=linewidth)
     plt.plot(time[0,:],targets[1,:],color="orange", linewidth=linewidth)
     plt.ylabel('Target \n functions', fontsize=fontsize)
     plt.xticks(fontsize=fontsize)
     plt.yticks(fontsize=fontsize)
     plt.ylim([-2,3.5])
     plt.legend(legend_entries,frameon=False)

     plt.subplot(312)
     plt.plot(time[0,:],z_outmat[0,:],color="red", linewidth=linewidth)
     plt.plot(time[0,:],z_outmat[1,:],color="orange", linewidth=linewidth)
     plt.ylabel('Output', fontsize=fontsize)
     plt.xticks(fontsize=fontsize)
     plt.yticks(fontsize=fontsize)
     plt.ylim([-2,3.5])
     plt.legend(legend_entries,frameon=False)

     plt.subplot(313)
     plt.plot(time[0,:],learning_error[0,:],color="darkred", linewidth=linewidth)
     plt.plot(time[0,:],learning_error[1,:],color="coral", linewidth=linewidth)
     plt.ylabel('Learning \n error', fontsize=fontsize)
     plt.xticks(fontsize=fontsize)
     plt.yticks(fontsize=fontsize)
     plt.legend(legend_entries,frameon=False)
     
     plt.suptitle(f"Memorized time series No.{i+1}")
     plt.tight_layout()

#%% 
def dynamics_separated(N,tau,delta,totaltime,J,r,x,z_out,w_out):
    for t in range(totaltime):
            r[:, t] = np.tanh(x).reshape(N,) # activation function to calculate rates
            z_out[:,t] = np.dot(w_out.T,r[:,t].reshape(N,)) # zi(t)=sum (Jij rj) over j
            x = x + (-x + np.dot(J,r[:,t]).reshape(N,1))*(delta/tau) # Euler update for activity x
    return z_out,w_out,x

#%% add gaussian noise
def add_noise(x,sigma):
    noise=np.random.normal(0,sigma,size=(np.shape(x)))
    x=x+noise
    return x

#%% train and test loop for trakr

## should work if there's only one output unit

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

#%% #%% train and test loop for trakr - when learning chaotic RNN unit trajectories - learning ...
#%%      multiple units simultaneously - dev project

## should work if there are multiple input/output units
def train_test_loop_formultiunitactivity(x_train,N,N_out,g,tau,delta,alpha,totaltime):  
    learning_error_tot=np.zeros((np.size(x_train,0),totaltime,np.size(x_train,2),np.size(x_train,2)))
    for i in range(np.size(x_train,2)): 
        regP=alpha*np.identity(N) # regularizer
        J = g*np.random.randn(N,N)/np.sqrt(N) # connectivity matrix J
        r = np.zeros((N, totaltime)) # rate matrix - firing rates of neurons
        x = np.random.randn(N, 1) # activity matrix before activation function applied
        z_out = np.zeros((N_out,totaltime)) # z(t) for the output read out unit
        error = np.zeros((N_out, totaltime)) # error signal- z(t)-f(t)
        learning_error = np.zeros((N_out, totaltime)) # change in the learning error over time
        w_out = np.random.randn(N, N_out)/np.sqrt(N) # output weights for the read out unit
        w_in = np.random.randn(N, N_out) # input weights
        f=x_train[:,:,i]
        # pdb.set_trace()
        error,learning_error,z_out,w_out,x,regP=dynamics(N_out,N,g,tau,delta,f,
                        totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                        freezew=0,t1_train=0,
                        t2_train=totaltime)
        print(i)
        for j in range(np.size(x_train,2)):
            f=x_train[:,:,j]
            error,learning_error,z_out,w_out,_,_=dynamics(N_out,N,g,tau,delta,f,
                        totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                        freezew=1,t1_train=0,
                        t2_train=totaltime)
            learning_error_tot[:,:,j,i]=learning_error
    #         print(j)
    return learning_error_tot


#%% cross validation metrics for trakr ; accuracy and auc
def cross_val_metrics_trakr(x,y,n_classes,splits):
    skf = StratifiedKFold(n_splits=splits)
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
# def generateMNISTdata():
#     (X_arr, y_arr), (
#         Xtest,
#         ytest,
#     ) = tf.keras.datasets.mnist.load_data()

#     X=np.zeros((10,100,28,28))
#     y=np.zeros((10,100))
#     for i in range(10):
#           tempx = X_arr[np.where((y_arr == i ))]
#           tempy= y_arr[np.where((y_arr == i ))]
#           ind = np.random.choice(np.size(tempx,0), size=100, replace=False)
#           X[i,:,:,:]=tempx[ind,:]
#           y[i,:]=tempy[ind]

#     x_test=X.reshape(1000,784)
#     y_test=y.reshape(1000)
#     return x_test,y_test


#%%

#################################
# Based on the code provided by Fawaz et al related to their 2019 paper.
# https://link.springer.com/article/10.1007/s10618-019-00619-1
# https://github.com/hfawaz/dl-4-tsc
#################################
# import twiesn

# def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
#     return twiesn.Classifier_TWIESN(output_directory, verbose=True)



#%% visualize the training with trakr

def visualize_training(f,N,N_out,g,tau,delta,alpha):
    totaltime=np.size(f,1)
    regP=alpha*np.identity(N) # regularizer
    J = g*np.random.randn(N,N)/np.sqrt(N) # connectivity matrix J
    r = np.zeros((N, totaltime)) # rate matrix - firing rates of neurons
    x = np.random.randn(N, 1) # activity matrix before activation function applied
    z_out = np.zeros((N_out,totaltime)) # z(t) for the output read out unit
    error = np.zeros((N_out, totaltime)) # error signal- z(t)-f(t)
    learning_error = np.zeros((N_out, totaltime)) # change in the learning error over time
    w_out = np.random.randn(N, N_out)/np.sqrt(N) # output weights for the read out unit
    w_in = np.random.randn(N, N_out) # input weights
    # pdb.set_trace()
    error,learning_error,z_out,w_out,x,regP=dynamics(N_out,N,g,tau,delta,f,
                    totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                    freezew=0,t1_train=0,
                    t2_train=totaltime/3)
    return f,z_out,learning_error



#%% standardize data 

def standardize_data(dataorig):
    data=dataorig.reshape(np.size(dataorig,0)*np.size(dataorig,1),1)
    data=stats.zscore(data)
    data=data.reshape(np.size(dataorig,0),np.size(dataorig,1))
    return data


#%% notch filter different frequencies

def filt_notch(dataorig,fs,f0):
    Q = 30.0  # Quality factor
    w0 = f0 / (fs / 2 )  # Normalized Frequency
    b, a = signal.iirnotch( w0, Q )   
    data = signal.filtfilt(b, a, dataorig.reshape(1,np.size(dataorig,0)*np.size(dataorig,1)))
    data=data.reshape(np.size(dataorig,0),np.size(dataorig,1))
    return data

#%% band pass filter between different frequencies

def band_pass_filt(dataorig,bands):
    data=dataorig.reshape(1,np.size(dataorig,0)*np.size(dataorig,1))
    sos=signal.butter(4, bands, btype='bandpass', output='sos')
    data = sosfiltfilt(sos, data)
    data=data.reshape(np.size(dataorig,0),np.size(dataorig,1))
    return data