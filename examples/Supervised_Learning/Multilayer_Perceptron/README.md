# Multilayer Perceptron (MLP)

## Overview
This notebook demonstrates a Multilayer Perceptron classifier implemented from scratch in `rice_ml`. The MLP extends the Perceptron by introducing hidden layers and non-linear activation functions, allowing it to learn complex decision boundaries.

## Example Walkthrough
1. Dataset description  
   Explanation and minor exploration and visualization of the data.

2. Preprocessing steps  
   Images are flattened into feature vectors and normalized to ensure stable gradient-based optimization.

3. Model training  
   The MLP is trained using backpropagation and gradient descent. Hidden layer size and learning rate are chosen explicitly to illustrate convergence behavior.

4. Evaluation metrics  
   Model performance is evaluated using classification accuracy and training loss curves.

## Failure Modes
MLPs are sensitive to hyperparameter choices and can overfit small datasets. Poor initialization or learning rates may prevent convergence.

## Data Used In Notebook
Handwritten digit dataset (MNIST-style), with images flattened into numeric feature vectors and multi-class labels.

## Comparison Notes (Optional)
Compared to logistic regression, the MLP can model non-linear decision boundaries but is less interpretable and requires more tuning.
