# Logistic Regression

## Overview
This notebook applies logistic regression to a binary classification task using the `rice_ml` implementation. Logistic regression models class probabilities using a linear decision boundary and a sigmoid activation.

## Example Walkthrough
1. Dataset description  
    Explanation and minor exploration and visualization of the data.

2. Preprocessing steps  
   Input features are normalized and the dataset is split into training and testing sets.

3. Model training  
   The model is trained using gradient descent to minimize the logistic loss.

4. Evaluation metrics  
   Classification accuracy and a confusion matrix are used to evaluate performance.

## Failure Modes
Logistic regression performs poorly when classes are not linearly separable or when important non-linear interactions are present.

## Data Used In Notebook
Handwritten digit dataset (MNIST-style), with labels converted to a binary classification task.

## Comparison Notes (Optional)
Logistic regression is simpler and more interpretable than MLPs, but lacks the ability to model complex non-linear patterns.
