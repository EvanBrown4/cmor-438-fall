# Perceptron

## Overview
This notebook demonstrates the Perceptron algorithm for binary classification using the from-scratch implementation in `rice_ml`. The Perceptron is a foundational linear classifier that updates weights iteratively based on misclassified examples.

## Example Walkthrough
1. Dataset description  
   Explanation and minor exploration and visualization of the data.

2. Preprocessing steps  
   Images are flattened into numeric feature vectors and normalized to improve convergence during training.

3. Model training  
   The Perceptron is trained using an online learning rule that updates weights whenever a sample is misclassified. Training proceeds over multiple epochs until convergence or a maximum iteration count is reached.

4. Evaluation metrics  
   Classification accuracy is used to assess model performance on held-out test data.

## Failure Modes
The Perceptron only converges when the data is linearly separable. It performs poorly on problems requiring non-linear decision boundaries and is sensitive to feature scaling.

## Data Used In Notebook
Handwritten digit dataset (MNIST-style), converted into a binary classification problem with flattened pixel features.

## Comparison Notes (Optional)
The Perceptron is simpler and more interpretable than logistic regression but does not provide probabilistic outputs. It also serves as the conceptual foundation for multilayer neural networks.
