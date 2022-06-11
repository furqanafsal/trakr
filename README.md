## TRAKR

This repo contains basic code to train, test and classify time series patterns using TRAKR.

For details, see: Reservoir-based Tracking (TRAKR) For One-shot Classification Of Neural Time-series Patterns

[https://www.biorxiv.org/content/10.1101/2021.10.13.464288v1](https://www.biorxiv.org/content/10.1101/2021.10.13.464288v3)

##

I also reuse/ reimplement some code from Fawaz et al 2019 related to their paper and repo as below:

 https://link.springer.com/article/10.1007/s10618-019-00619-1 
 
 https://github.com/hfawaz/dl-4-tsc

##

We provide Python implementations here. 

All software distributed under GNU GPL (see LICENSE.md).

A description of different files within this repository is provided below:

## Files

- <constants.py> - reused from Fawaz et al 2019

- <dtw.py> - contains an implementation of DTW from the following paper:

Thanawin Rakthanmanon, Bilson Campana, Abdullah Mueen, Gustavo Batista, Brandon Westover, Qiang Zhu, Jesin Zakaria, and Eamonn Keogh. 2012. Searching and mining trillions of time series subsequences under dynamic time warping. In Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining (KDD '12). Association for Computing Machinery, New York, NY, USA, 262–270. DOI:https://doi.org/10.1145/2339530.2339576

- <esn.py> - implementation based on twiESN from Fawaz et al 2019 

- <euclidean.py> - Euclidean distance for the classification of time series patterns

- <main.py> - main TRAKR file for the classification of time series patterns

- <mlp.py> - MLP for the classification of patterns - based on Fawaz et al 2019

- <modules.py> - custom functions used for different types of training, metric calculations etc

- <mutualinfo.py> - implements mutual info for the classification of patterns

- <naivebayes.py> - implements Naive Bayes for classification

- <trakr_patternchange_syntheticsignals.py> - recreate Figure 2 from biorxiv paper

- <trakr_simple.py> - simple version of trakr to train and test a single sine wave 

- <twiesn.py> - reused/modified from Fawaz et al 2019

- <utils.py> - reused from Fawaz et al 2019

## Requirements

- The following versions were used for most of the code.

Python 3.7.4

Keras                              2.4.3

matplotlib                         3.1.1

numpy                              1.17.2

scikit-learn                       0.23.1

scipy                              1.4.1

spyder-kernels                     2.2.0

tensorflow                         2.3.0



##
- For DTW, the following were used.

Python            3.5.6 

numpy             1.13.3 

scikit-learn      0.22.2.post1

scipy             1.4.1   

spyder-kernels    2.2.0  

ucrdtw            0.0.0  












