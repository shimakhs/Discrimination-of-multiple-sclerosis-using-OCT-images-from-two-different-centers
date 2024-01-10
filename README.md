# Discrimination of multiple sclerosis using OCT images from two different centers

If you use these codes please cite to our paper: https://www.sciencedirect.com/science/article/pii/S2211034823003486 
 
This is a binary classification of :

Multiple Sclerosis (MS)
Healthy Controls (HC)

Using Machine learning algorithms:
Support Vector Machine (SVM)
Random Forest
Artificial Neural Network

Makes the classification results Interpretable by making Heat-Map


OctRead.py
To read .vol OCT dataset

Thickness_function.py:

It is a function that when is called:

1. read a .vol data
2. flip left eyes to right ones
3. Compute thicknesses as distance between boundary points
4. Rotate and flip to align with fundus
5. Make ThicknessMaps
6. resize ThicknessMaps

main_SVM.py:

By calling the Thickness_function.py makes the vectorized thicknessMaps 
Then it apply SVM classifier using PCA for dimenstion reducation and 10-fold cross validation method 
Finally calculates the Accuracy, Precision, Recall, F1-score and confusion matrix to evaluate the results

main_RF.py:

By calling the Thickness_function.py makes the vectorized thicknessMaps 
Then it apply RF classifier using grid-search method to find appropriate parameters and and 10-fold cross validation method 
Finally calculates the Accuracy to evaluate the results

heat-map.py: ### Using Occlusion Sensitivity

By calling the Thickness_function.py makes the vectorized thicknessMaps 
Then:
1. Train the SVM classifier using train set dataset
2. A black mask with the size of 10×10 pixels moves on the test set with a single step to sweep the whole image. (The locations of the pixels covered by the mask are transferred to vector-shaped positions) 
3. The masked vector is sent as input to the model and the accuracy is calculated.
4. The interpretability is shown by regenerating the occlusion with the original image size, with the value of accuracy in the location of each pixel (called the heat map)





