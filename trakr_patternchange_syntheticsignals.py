from scipy.signal import chirp
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
from sklearn.metrics import mean_squared_error
from sklearn import decomposition
from pdb import set_trace
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import svd,subspace_angles
from modules import dynamics

plt.rcParams['figure.figsize'] = [6, 4]
plt.rcParams['figure.dpi'] = 300
font=14
linewidth=4

totaltime=5000 # length of synthetic signal
time = np.linspace(0, 2,totaltime).reshape(1,totaltime) # time vector

amp=np.array([1,1,1])  # amplitude of synthetic signal
freq=np.array([10,20,15]) # frequency array

# train signal
f=np.concatenate((amp[0]*np.sin(2*np.pi* freq[0] * np.linspace(0, 1,totaltime/2)) , 
                  amp[1]*np.sin(2*np.pi* freq[1] * np.linspace(0, 1, totaltime/2))
                 )).reshape(1,totaltime) # target function

#test signal
ftest=np.concatenate((amp[0]*np.sin(2*np.pi* freq[1] * np.linspace(0, 1,totaltime/2)) , 
                  amp[1]*np.sin(2*np.pi* freq[0]* np.linspace(0, 1, totaltime/2))
                 
                 )).reshape(1,totaltime) 


N_out=1
N=30 # number of neurons in the RNN
g=1.2 # gain
tau=1 # tau
delta = .1 # delta for Euler's method
alpha=1 # alpha for regularizer
regP=alpha*np.identity(N) # regularizer
J = g*np.random.randn(N,N)/np.sqrt(N) # connectivity matrix J
r = np.zeros((N, totaltime)) # rate matrix - firing rates of neurons
x = np.random.randn(N, 1) # activity matrix before activation function applied
z_out = np.zeros((N_out,totaltime)) # z(t) for the output read out unit
error = np.zeros((N_out, totaltime)) # error signal- z(t)-f(t)
learning_error = np.zeros((N_out, totaltime)) # change in the learning error over time
w_out = np.random.randn(N, N_out)/np.sqrt(N) # output weights for the read out unit
w_in = np.random.randn(N, N_out) # input weights 



    
#train - realtime
    
error,learning_error,z_out,w_out,x,regP=dynamics(N_out,N,g,tau,delta,f,
                totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
                freezew=0,t1_train=totaltime/2,
                t2_train=totaltime)



# test - realtime
error,learning_error,z_out,w_out,_,_=dynamics(N_out,N,g,tau,delta,ftest,
            totaltime,regP,J,r,x,z_out,error,learning_error,w_out,w_in,
            freezew=1,t1_train=0,
            t2_train=totaltime)


num=0

##Plotting train signal, test signal and error
%matplotlib inline

plt.figure()
ax=plt.subplot(411)
ax.plot(time[0,num:],f[0,num:],color="blue", linewidth=linewidth,label='Train Signal') 
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.ylim([-1,1])
plt.ylabel('Train Signal', fontsize=font)
ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.yticks(np.arange(-1, 1.1, 1))

ax=plt.subplot(412)
ax.plot(time[0,num:],ftest[0,num:],color="orange", linewidth=linewidth,label='Test Signal') 
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.ylim([-1,1])
plt.ylabel('Test Signal', fontsize=font)
ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.yticks(np.arange(-1, 1.1, 1))

ax=plt.subplot(413)
ax.plot(time[0,num:],z_out[0,num:],color="red", linewidth=linewidth,label=r'Network Output, $z_{out}(t)$')
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
plt.ylim([-1,1])
plt.ylabel(r"$z_{out}(t)$", fontsize=font)
ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.yticks(np.arange(-1, 1.1, 1))

ax=plt.subplot(414)
ax.plot(time[0,num:],learning_error[0,num:]/np.max(learning_error[0,num:]),color="green", linewidth=linewidth,label='E(t)')
#plt.ylim([0,0.1])
plt.xlabel('Time (s)', fontsize=font)
plt.ylabel('E(t)', fontsize=font)
plt.xticks(fontsize=font)
plt.yticks(fontsize=font)
#ax.xaxis.set_major_formatter(plt.NullFormatter())
plt.yticks(np.arange(0, 1.1, .5))
