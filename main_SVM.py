# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 10:36:02 2022

Author: Shima
Description:
This script performs feature extraction and classification of Multiple Sclerosis (MS)
and Healthy Controls (HC) using OCT images. Features are extracted from .vol files,
processed into GCIPL thickness maps, and classified using PCA and SVM.

Inputs:
- Directory containing .vol files for MS dataset (path1)
- Directory containing .vol files for HC dataset (path2)

Outputs:
- Combined dataset of processed GCIPL thickness maps with labels (MS: 1, HC: 0)
- Classification performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrix of predictions
"""

# ---------------------- Imports ----------------------
# Importing necessary libraries and modules for data processing, visualization, 
# machine learning, and OCT image feature extraction
import scipy
import tensorflow as tf
import keras
import numpy as np
from scipy import signal, misc
from struct import unpack
import os
import math
import matplotlib.pyplot as plt
from itertools import permutations
import cv2
from skimage import data, io, filters
from skimage.transform import rotate
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, make_scorer
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import Thickness_func
from OctRead import OctRead  # Module for reading .vol OCT dataset files

# -------------------- Dataset Paths --------------------
# Define directories containing MS and HC .vol datasets
path1 = 'The directory of .vol of MS dataset'  # Path to MS dataset
path2 = 'The directory of .vol of HC dataset'  # Path to HC dataset

dirs = os.listdir(path1)  # List of files in MS directory
dirs2 = os.listdir(path2)  # List of files in HC directory

# ------------------- Initialize DataFrames -------------------
# Initialize DataFrames to store processed thickness features
Final_results_GCIPL_40_MS = pd.DataFrame(np.zeros((len(dirs), 1601)))
Final_results_GCIPL_40_HC = pd.DataFrame(np.zeros((len(dirs2), 1601)))

# ------------------ Feature Extraction ------------------
# Process MS dataset
for n in range(len(dirs)):
    file_name = os.path.join(path1, dirs[n])  # Full path to the current .vol file
    
    # Extract thickness features and reshape them to a 1D array
    thicknessVals_resized_GCIPL_40_MS = Thickness_func.feature(file_name).reshape(1, 40 * 40)
    
    # Store features in the DataFrame
    Final_results_GCIPL_40_MS.iloc[n, 0:1600] = thicknessVals_resized_GCIPL_40_MS
    
    # Assign filename as the index and label the sample as MS (1)
    Final_results_GCIPL_40_MS = Final_results_GCIPL_40_MS.rename(index={n: dirs[n]})
    Final_results_GCIPL_40_MS.iloc[n, 1600] = 1

# Process HC dataset
for n in range(len(dirs2)):
    file_name = os.path.join(path2, dirs2[n])  # Full path to the current .vol file
    
    # Extract thickness features and reshape them to a 1D array
    thicknessVals_resized_GCIPL_40_HC = Thickness_func.feature(file_name).reshape(1, 40 * 40)
    
    # Store features in the DataFrame
    Final_results_GCIPL_40_HC.iloc[n, 0:1600] = thicknessVals_resized_GCIPL_40_HC
    
    # Assign filename as the index and label the sample as HC (0)
    Final_results_GCIPL_40_HC = Final_results_GCIPL_40_HC.rename(index={n: dirs2[n]})
    Final_results_GCIPL_40_HC.iloc[n, 1600] = 0

# Combine MS and HC DataFrames
Final_results_GCIPL_40 = pd.concat([Final_results_GCIPL_40_MS, Final_results_GCIPL_40_HC], ignore_index=True)

# -------------------- Prepare Features and Labels --------------------
# Extract features (X) and labels (y)
data = Final_results_GCIPL_40
X = data.iloc[:, 1:1601]  # Features (GCIPL thickness maps)
y = data.iloc[:, 1600:1601]  # Labels (1: MS, 0: HC)

# --------------------- Standardization ---------------------
# Standardize features for PCA and SVM
sc = StandardScaler()
X_train_std = sc.fit_transform(X)

# --------------------- PCA Dimensionality Reduction ---------------------
n_components = 350  # Number of PCA components
pca = PCA(n_components=n_components)
pca.fit(X_train_std)
train_img = pca.transform(X_train_std)

# --------------------- SVM Classification ---------------------
# Initialize SVM classifier with a linear kernel
clf = SVC(kernel='linear')

# Define scoring metrics for cross-validation
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1_score': make_scorer(f1_score),
}

# Perform 10-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=0)
results = model_selection.cross_validate(
    estimator=clf,
    X=train_img,
    y=y,
    cv=kfold,
    scoring=('accuracy', 'precision', 'recall', 'f1')
)

# Calculate average metrics
test_accuracy = np.mean(results['test_accuracy'])
test_precision = np.mean(results['test_precision'])
test_recall = np.mean(results['test_recall'])
test_f1score = np.mean(results['test_f1'])

# --------------------- Confusion Matrix ---------------------
# Perform cross-validation predictions
y_pred = cross_val_predict(clf, X=train_img, y=y, cv=kfold)

# Compute confusion matrix
conf_mat = confusion_matrix(y, y_pred)

# --------------------- Output Results ---------------------
# Print evaluation metrics and confusion matrix
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1score}")
print("Confusion Matrix:")
print(conf_mat)
