# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 13:19:01 2022

Author: Shima

Description:
This script performs feature extraction and classification for OCT images 
to differentiate between Multiple Sclerosis (MS) and Healthy Controls (HC). 
It uses GCIPL thickness features, applies Support Vector Machine (SVM) classifiers, 
and evaluates feature importance via accuracy heatmaps.

Inputs:
- `path1`: Directory containing .vol files for the MS dataset.
- `path2`: Directory containing .vol files for the HC dataset.

Outputs:
- Combined dataset with GCIPL thickness features and labels.
- Accuracy heatmaps indicating feature importance for classification.
- Classification report for the SVM models.
"""

# -------------------- Imports --------------------
import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import Thickness_func  # Custom module for feature extraction
from OctRead import OctRead  # Module for reading .vol OCT dataset files

# -------------------- Input Directories --------------------
# Paths to the MS and HC datasets (update with actual paths)
path1 = 'The directory of .vol of MS dataset'  # Example: 'datasets/MS/'
path2 = 'The directory of .vol of HC dataset'  # Example: 'datasets/HC/'

# List of all files in the directories
dirs = os.listdir(path1)
dirs2 = os.listdir(path2)

# -------------------- Initialize DataFrames --------------------
# DataFrames to store GCIPL thickness features and corresponding labels
Final_results_GCIPL_40_MS = pd.DataFrame(np.zeros((len(dirs), 1601)))  # MS dataset
Final_results_GCIPL_40_HC = pd.DataFrame(np.zeros((len(dirs2), 1601)))  # HC dataset

# -------------------- Feature Extraction --------------------
# Process MS dataset
for n, file_name in enumerate(dirs):
    file_path = os.path.join(path1, file_name)
    thickness_map = Thickness_func.feature(file_path).reshape(1, 40 * 40)  # Extract features
    Final_results_GCIPL_40_MS.iloc[n, :-1] = thickness_map  # Store features
    Final_results_GCIPL_40_MS.rename(index={n: file_name}, inplace=True)
    Final_results_GCIPL_40_MS.iloc[n, -1] = 1  # Label for MS (1)

# Process HC dataset
for n, file_name in enumerate(dirs2):
    file_path = os.path.join(path2, file_name)
    thickness_map = Thickness_func.feature(file_path).reshape(1, 40 * 40)  # Extract features
    Final_results_GCIPL_40_HC.iloc[n, :-1] = thickness_map  # Store features
    Final_results_GCIPL_40_HC.rename(index={n: file_name}, inplace=True)
    Final_results_GCIPL_40_HC.iloc[n, -1] = 0  # Label for HC (0)

# Combine MS and HC datasets
Final_results_GCIPL_40 = pd.concat([Final_results_GCIPL_40_MS, Final_results_GCIPL_40_HC], ignore_index=True)

# -------------------- Data Preparation --------------------
# Separate features (X) and labels (y)
X = Final_results_GCIPL_40.iloc[:, :-1]  # GCIPL thickness features
y = Final_results_GCIPL_40.iloc[:, -1]  # Labels (MS: 1, HC: 0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# -------------------- SVM Classifier --------------------
# Define SVM kernel types
kernels = ['Polynomial', 'RBF', 'Sigmoid', 'Linear']

# Function to get the SVM model based on the kernel type
def getClassifier(ktype):
    if ktype == 0:  # Polynomial kernel
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:  # Radial Basis Function (RBF) kernel
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:  # Sigmoid kernel
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:  # Linear kernel
        return SVC(kernel='linear', gamma="auto")

# Initialize the classifier (using Linear kernel in this case)
svclassifier = getClassifier(3)
svclassifier.fit(X_train, y_train)  # Train the classifier

# -------------------- Heatmap Evaluation --------------------
# Initialize heatmaps to store accuracy values
heat_acc = np.zeros((31, 31))  # First region
heat_acc2 = np.zeros((31, 31))  # Second region

# Evaluate feature importance by perturbing test data
for k in range(31):
    for j in range(31):
        # Perturb features in the first region (rows 0-10)
        X_test_zero = X_test.copy()
        for i in range(10):
            X_test_zero.iloc[:, (40 * (i + k) + j):(40 * (i + k) + 10 + j)] = 0
        y_pred = svclassifier.predict(X_test_zero)  # Predict perturbed data
        heat_acc[k, j] = metrics.accuracy_score(y_test, y_pred)  # Store accuracy

        # Perturb features in the second region (rows 40-50)
        X_test_zero2 = X_test.copy()
        for i in range(40, 50):
            X_test_zero2.iloc[:, (40 * (i + k) + j):(40 * (i + k) + 10 + j)] = 0
        y_pred2 = svclassifier.predict(X_test_zero2)  # Predict perturbed data
        heat_acc2[k, j] = metrics.accuracy_score(y_test, y_pred2)  # Store accuracy

# -------------------- Results --------------------
# The `heat_acc` and `heat_acc2` arrays represent the accuracy heatmaps
# These heatmaps indicate the importance of different features in classification

print("Heatmap evaluation completed. Results are stored in 'heat_acc' and 'heat_acc2'.")
