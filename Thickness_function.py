# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 09:07:53 2022

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

from OctRead import OctRead


def feature(file_name):
   myclass = OctRead(file_name)
   octhdr = myclass.get_oct_hdr()
   thicknessGrid = myclass.get_thickness_grid(octhdr)
   sloImage = myclass.get_slo_image(octhdr)
   BScans = myclass.get_b_scans(octhdr)
   segmentations = myclass.get_segmentation(octhdr)
   BScans[BScans>1e+38] = 0
   
   if  str(octhdr['ScanPosition'])[2:4] in 'OS':
      sloImage = np.fliplr(sloImage)
      
   ######## show Boundaries on Bscan
   Boudaries = segmentations['SegLayers']
   
 
######################  Thicknessmap #####################

   Boudaries_0 = Boudaries[:,0,:]
   Boudaries_2 = Boudaries[:,2,:]
   Boudaries_4 = Boudaries[:,4,:]
   Boudaries_5 = Boudaries[:,5,:]
   Boudaries_6 = Boudaries[:,6,:]
   Boudaries_8 = Boudaries[:,8,:]
   Boudaries_14 = Boudaries[:,14,:]
   Boudaries_15 = Boudaries[:,15,:]
   Boudaries_1 = Boudaries[:,1,:]

   bd_pts = np.zeros((BScans.shape[1],octhdr['NumBScans'],9))
   bd_pts[:,:,0] = Boudaries_0
   bd_pts[:,:,1] = Boudaries_2
   bd_pts[:,:,2] = Boudaries_4
   bd_pts[:,:,3] = Boudaries_5
   bd_pts[:,:,4] = Boudaries_6
   bd_pts[:,:,5] = Boudaries_8
   bd_pts[:,:,6] = Boudaries_14
   bd_pts[:,:,7] = Boudaries_15
   bd_pts[:,:,8] = Boudaries_1

 ########## Compute thickness as distance between boundary points
   boundaryPoints= bd_pts
   thickness= np.diff(boundaryPoints,1,2)
   thicknessVals = np.squeeze(thickness)*octhdr['ScaleZ']*1000
   
   
   #### Rotate and flip to align with fundus


   thicknessVals = np.flip(thicknessVals,0)
   thicknessVals = np.flip(thicknessVals,1)
########## Flip if left eye for consistency of 'nasal' to the right side
   if not str(octhdr['ScanPosition'])[2:4] in 'OD':
       thicknessVals = np.flip(thicknessVals,1)   
       
   thicknessVals_resized_RNFL_40 = cv2.resize(thicknessVals[:,:,0],(40,40))
   thicknessVals_resized_GCIPL_40 = cv2.resize(thicknessVals[:,:,1],(40,40))

   return thicknessVals_resized_GCIPL_40


















