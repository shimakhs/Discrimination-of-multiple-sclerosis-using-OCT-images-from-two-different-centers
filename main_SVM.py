# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:36:02 2022

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

# standardize the features
sc = StandardScaler()
X_train_std = sc.fit_transform(X)

n_components = 350

# Make an instance of the Model
pca = PCA(n_components)
pca.fit(X_train_std)

train_img = pca.transform(X_train_std)

clf = svm.SVC(kernel='linear')


scoring = {'accuracy' : make_scorer(accuracy_score), 
           'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score)}
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, shuffle = True, random_state = 0)
results = model_selection.cross_validate(estimator=clf,
                                          X=train_img,
                                          y=y,
                                          cv=kfold,
                                          scoring=('accuracy','precision','recall','f1'))


test_accuracy = np.mean(results['test_accuracy'])
test_precision = np.mean(results['test_precision'])
test_recall = np.mean(results['test_recall'])
test_f1score = np.mean(results['test_f1'])


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_pred = cross_val_predict(clf, X=train_img, y=y, cv=kfold)
conf_mat = confusion_matrix(y, y_pred)








