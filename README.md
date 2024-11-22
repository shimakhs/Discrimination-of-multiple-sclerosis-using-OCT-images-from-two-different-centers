# Discrimination of multiple sclerosis using OCT images from two different centers

This repository contains the code for binary classification of **Multiple Sclerosis (MS)** and **Healthy Controls (HC)** using **Optical Coherence Tomography (OCT)** images. The project utilizes various machine learning algorithms and provides interpretable results through heat maps.

If you use this code, please cite our paper:  
[Discrimination of multiple sclerosis using OCT images from two different centers](https://www.sciencedirect.com/science/article/pii/S2211034823003486).

## Overview
The project focuses on:
- **classification**: Distinguishing MS patients from healthy controls.  
- **Machine learning algorithms**:
  - Support Vector Machine (SVM)
  - Random Forest (RF)
  - Artificial Neural Network (ANN)  
- **Interpretability**: Generating heat maps to visualize model decisions.

---

## Repository Contents
### 1. **`OctRead.py`**
Reads `.vol` OCT dataset files.

---

### 2. **`Thickness_function.py`**
A utility function that processes OCT images to generate thickness maps. It performs the following tasks:
1. Reads `.vol` data.
2. Flips left-eye images to align with right-eye images.
3. Computes thicknesses as the distance between boundary points.
4. Rotates and flips images to align with fundus views.
5. Generates and resizes thickness maps.

---

### 3. **`main_SVM.py`**
Implements the SVM classification pipeline:
1. Calls `Thickness_function.py` to generate vectorized thickness maps.
2. Applies Principal Component Analysis (PCA) for dimensionality reduction.
3. Trains an SVM classifier using a 10-fold cross-validation method.
4. Evaluates results using metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix

---

### 4. **`main_RF.py`**
Implements the Random Forest classification pipeline:
1. Calls `Thickness_function.py` to generate vectorized thickness maps.
2. Uses grid search to identify optimal RF parameters.
3. Trains the RF classifier using a 10-fold cross-validation method.
4. Evaluates results based on accuracy.

---

### 5. **`heat-map.py`**
Generates interpretability results using occlusion sensitivity:
1. Calls `Thickness_function.py` to generate vectorized thickness maps.
2. Trains an SVM classifier on the training dataset.
3. Applies a moving black mask (10×10 pixels) to the test set to sweep across the image.
4. Calculates accuracy for each masked region.
5. Regenerates a heat map by mapping accuracy values back to the original image size.

---

## Usage
1. Preprocess OCT data using `OctRead.py` and `Thickness_function.py`.
2. Train and evaluate classifiers using:
   - `main_SVM.py` for SVM.
   - `main_RF.py` for Random Forest.
3. Visualize results and interpretability using `heat-map.py`.

