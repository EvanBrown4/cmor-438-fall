# K-Nearest Neighbors (KNN)

## Overview
This notebook demonstrates the K-Nearest Neighbors (KNN) algorithm for both classification and regression tasks using the `rice_ml` implementation. KNN is a non-parametric, instance-based method that makes predictions based on proximity in feature space.

## Example Walkthrough
1. Dataset description  
   Two datasets are used: a handwritten digit dataset for classification and a housing dataset for regression. This section consists of an explanation and minor exploration and visualization of the data.


2. Preprocessing steps  
   All features are normalized prior to training to ensure that distance calculations are meaningful. For the digit dataset, images are flattened into feature vectors.

3. Model training  
   The KNN model stores the training data and makes predictions by selecting the k closest neighbors under a chosen distance metric. Different values of k are explored to study bias–variance trade-offs.

4. Evaluation metrics  
   Classification performance is evaluated using accuracy, while regression performance is evaluated using RMSE.

## Failure Modes
KNN performs poorly on high-dimensional datasets due to the curse of dimensionality and becomes computationally expensive as dataset size grows. Model performance is also highly sensitive to feature scaling and the choice of k.

## Data Used In Notebook
- **Classification:** Handwritten digit dataset (MNIST-style), with images flattened into numeric feature vectors.  
- **Regression:** California Housing dataset, consisting of numeric demographic and geographic features used to predict median house values.

## Comparison Notes (Optional)
Compared to linear models, KNN makes fewer assumptions about data structure but scales poorly. Compared to tree-based methods, KNN requires careful preprocessing but can perform competitively on well-behaved datasets.
