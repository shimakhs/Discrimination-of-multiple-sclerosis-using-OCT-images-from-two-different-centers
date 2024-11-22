# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 12:59:12 2022

Author: Shima

Description:
This script performs feature extraction and classification of OCT images for 
Multiple Sclerosis (MS) and Healthy Controls (HC). The extracted GCIPL thickness 
features are processed and classified using a Random Forest model with nested 
cross-validation and hyperparameter tuning.

Inputs:
- path1: Directory containing .vol files for the MS dataset.
- path2: Directory containing .vol files for the HC dataset.

Outputs:
- Processed GCIPL thickness maps stored in a combined DataFrame with labels.
- Model performance metrics, including accuracy.
- Best hyperparameters of the Random Forest classifier.
"""

# -------------------- Imports --------------------
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import Thickness_func  # Custom module for feature extraction
from OctRead import OctRead  # Module for reading .vol OCT dataset files

# -------------------- Input Directories --------------------
# Paths to the MS and HC datasets (update with your actual paths)
path1 = 'The directory of .vol of MS dataset'  # Example: 'datasets/MS/'
path2 = 'The directory of .vol of HC dataset'  # Example: 'datasets/HC/'

# List of all files in the directories
dirs = os.listdir(path1)
dirs2 = os.listdir(path2)

# -------------------- Initialize DataFrames --------------------
# Create DataFrames to store extracted features and their labels
Final_results_GCIPL_40_MS = pd.DataFrame(np.zeros((len(dirs), 1601)))  # For MS
Final_results_GCIPL_40_HC = pd.DataFrame(np.zeros((len(dirs2), 1601)))  # For HC

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

# Combine MS and HC datasets into a single DataFrame
Final_results_GCIPL_40 = pd.concat([Final_results_GCIPL_40_MS, Final_results_GCIPL_40_HC], ignore_index=True)

# -------------------- Prepare Data --------------------
# Separate features (X) and labels (y)
X = Final_results_GCIPL_40.iloc[:, :-1]  # GCIPL thickness maps
y = Final_results_GCIPL_40.iloc[:, -1]  # Labels (MS: 1, HC: 0)

# -------------------- Nested Cross-Validation --------------------
# Define the outer cross-validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=0)

# Initialize a list to store accuracy results
outer_results = []

# Outer loop: Split data into training and testing sets
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :]
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

    # Define the inner cross-validation procedure
    cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)

    # Define the Random Forest model
    model = RandomForestClassifier(random_state=1)

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [10, 50, 100],  # Number of trees in the forest
        'max_features': [2, 4, 6]  # Number of features to consider at each split
    }

    # Perform a grid search with cross-validation
    search = GridSearchCV(model, param_grid, scoring='accuracy', cv=cv_inner, refit=True)
    result = search.fit(X_train, y_train)

    # Retrieve the best model and evaluate it on the test set
    best_model = result.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    outer_results.append(acc)  # Store accuracy

    # Report progress
    print(f'> Accuracy={acc:.3f}, Best Score={result.best_score_:.3f}, Params={result.best_params_}')

# -------------------- Output Results --------------------
# Summarize the performance of the model
mean_acc = np.mean(outer_results)
std_acc = np.std(outer_results)
print(f'Nested CV Accuracy: {mean_acc:.3f} ± {std_acc:.3f}')
