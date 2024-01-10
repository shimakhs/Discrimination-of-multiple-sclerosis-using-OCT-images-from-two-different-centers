# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 13:19:01 2022

@author: Shima
"""

####### Imports #############
import scipy
import tensorflow as tf
import keras
import numpy as np
import scipy as sp
import scipy.misc as misc
from scipy import signal
from scipy import misc
from struct import unpack
import os
import math
import matplotlib.pyplot as plt
from itertools import permutations 
import itertools
import os, glob
import cv2 
from os import listdir
from os.path import isfile, join
from skimage import data, io, filters 
#import pynput   
#from pynput.mouse import Controller
from PIL import Image
from skimage.transform import rotate
import skimage
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
#from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix 
from keras import metrics
from sklearn.metrics import classification_report
from keras.utils.np_utils import to_categorical
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,confusion_matrix
import Thickness_func

##### Import OctRead for reading the .vol dataset ###########

from OctRead import OctRead



path1 = 'The directory of .vol of MS dataset'
path2 = 'The directory of .vol of HC dataset'


dirs = os.listdir( path1 )
dirs2 = os.listdir( path2 )

count = 1

Final_results_GCIPL_40_MS = pd.DataFrame(np.zeros((len(dirs),1601)))
Final_results_GCIPL_40_HC = pd.DataFrame(np.zeros((len(dirs2),1601)))

#Final_results_GCIPL_40 = pd.DataFrame(np.zeros((len(dirs)+len(dirs2),1601)))

for n in range(0,len(dirs)):
    file_name=path1+'\\'+dirs[n]
   
    thicknessVals_resized_GCIPL_40_MS = Thickness_func.feature(file_name)
   
    Final_results_GCIPL_40_MS.iloc[n,0:1600] = thicknessVals_resized_GCIPL_40_MS.reshape(1,40*40)

    Final_results_GCIPL_40_MS = Final_results_GCIPL_40_MS.rename(index = {n : dirs[n]})

    Final_results_GCIPL_40_MS.iloc[n,1600] = 1  # for MS
   
   
    count += 1   
   

for n in range(0,len(dirs2)):
    file_name=path2+'\\'+dirs2[n]
   
    thicknessVals_resized_GCIPL_40_HC = Thickness_func.feature(file_name)
   
    Final_results_GCIPL_40_HC.iloc[n,0:1600] = thicknessVals_resized_GCIPL_40_HC.reshape(1,40*40)

    Final_results_GCIPL_40_HC = Final_results_GCIPL_40_HC.rename(index = {n : dirs2[n]})

    Final_results_GCIPL_40_HC.iloc[n,1600] = 0  # for HC
   
   
    count += 1      
   
Final_results_GCIPL_40 = pd.concat([Final_results_GCIPL_40_MS,Final_results_GCIPL_40_HC], ignore_index=True)   
   

data = Final_results_GCIPL_40   

data2=data.iloc[:,1:1601] ## GCIPL
X = data2
y = data.iloc[:,1600:1601]

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']

#A function which returns the corresponding SVC model
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(kernel='linear', gamma="auto")
target_names = ['class 0', 'class 1']

heat_acc = np.zeros((31,31))
heat_acc2 = np.zeros((31,31))

svclassifier = getClassifier(3) 
svclassifier.fit(X_train, y_train)# Make prediction


for k in range(0,31):
    for j in range(0,31):

       X_test_zero = X_test.copy()
       for i in range(0,10):
           X_test_zero.iloc[:,((40*(i+k))+j):((40*(i+k))+10+j)]=0
           y_pred = svclassifier.predict(X_test_zero)# Evaluate our model
           accuracy = (metrics.accuracy_score(y_test, y_pred))
       heat_acc[k,j] = accuracy 
for k in range(0,31):
    for j in range(0,31):

       X_test_zero2 = X_test.copy()
       for i in range(40,50):
           X_test_zero2.iloc[:,((40*(i+k))+j):((40*(i+k))+10+j)]=0
           y_pred2 = svclassifier.predict(X_test_zero2)# Evaluate our model
           accuracy = (metrics.accuracy_score(y_test, y_pred2))
       heat_acc2[k,j] = accuracy        
       



