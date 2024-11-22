# -*- coding: utf-8 -*-
"""
Created on Tue Nov 1 09:07:53 2022

Author: Shima

Description:
This script defines a feature extraction function for OCT images using the OctRead library. 
The extracted features include resized GCIPL (Ganglion Cell Inner Plexiform Layer) thickness maps.

Inputs:
- `file_name` (str): Path to the `.vol` file containing the OCT data.

Outputs:
- `thicknessVals_resized_GCIPL_40` (np.array): A 40x40 resized GCIPL thickness map.
"""

# -------------------- Imports --------------------
import numpy as np
import cv2
from OctRead import OctRead  # Import OctRead library for handling OCT .vol files

# -------------------- Feature Extraction Function --------------------
def feature(file_name):
    """
    Extracts features from an OCT `.vol` file and generates a resized GCIPL thickness map.

    Parameters:
        file_name (str): Path to the `.vol` file.

    Returns:
        np.array: 40x40 resized GCIPL thickness map.
    """
    # Initialize OctRead object
    myclass = OctRead(file_name)
    
    # Extract data from the .vol file
    octhdr = myclass.get_oct_hdr()  # Get OCT header
    thicknessGrid = myclass.get_thickness_grid(octhdr)  # Get thickness grid
    sloImage = myclass.get_slo_image(octhdr)  # Get SLO image
    BScans = myclass.get_b_scans(octhdr)  # Get B-Scans
    segmentations = myclass.get_segmentation(octhdr)  # Get segmentation boundaries
    
    # Handle invalid values in B-Scans
    BScans[BScans > 1e+38] = 0

    # Flip the SLO image for left-eye scans for consistency
    if str(octhdr['ScanPosition'])[2:4] in 'OS':  # Check if eye is "OS" (left eye)
        sloImage = np.fliplr(sloImage)

    # -------------------- Extract Boundary Points --------------------
    # Extract specific boundaries from the segmentation data
    boundaries = segmentations['SegLayers']
    boundaries_to_extract = [0, 2, 4, 5, 6, 8, 14, 15, 1]
    
    # Create a 3D array to store extracted boundaries
    bd_pts = np.zeros((BScans.shape[1], octhdr['NumBScans'], len(boundaries_to_extract)))
    for idx, boundary_index in enumerate(boundaries_to_extract):
        bd_pts[:, :, idx] = boundaries[:, boundary_index, :]

    # -------------------- Compute Thickness --------------------
    # Compute thickness as the distance between consecutive boundaries
    thickness = np.diff(bd_pts, axis=2)  # Difference between adjacent boundaries
    thicknessVals = np.squeeze(thickness) * octhdr['ScaleZ'] * 1000  # Convert to micrometers

    # -------------------- Align Thickness Map --------------------
    # Flip the thickness map to align it with the fundus image
    thicknessVals = np.flip(thicknessVals, axis=0)
    thicknessVals = np.flip(thicknessVals, axis=1)
    
    # If left eye (OS), flip horizontally for consistency (nasal on the right side)
    if str(octhdr['ScanPosition'])[2:4] not in 'OD':  # If not "OD" (right eye)
        thicknessVals = np.flip(thicknessVals, axis=1)

    # -------------------- Resize and Extract GCIPL Thickness --------------------
    # Resize GCIPL thickness map to 40x40 pixels
    thicknessVals_resized_GCIPL_40 = cv2.resize(thicknessVals[:, :, 1], (40, 40))  # GCIPL layer (index 1)

    return thicknessVals_resized_GCIPL_40

