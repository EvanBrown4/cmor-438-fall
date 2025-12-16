# Linear Regression

## Overview
This notebook demonstrates linear regression implemented from scratch in `rice_ml` for predicting continuous target values. The example emphasizes interpretability, baseline performance, and understanding model assumptions.

## Example Walkthrough
1. Dataset description  
    This section consists of an explanation and minor exploration of the data.
2. Preprocessing steps  
   Features are normalized prior to training to ensure stable optimization. The dataset is split into training and testing sets.

3. Model training  
   The linear regression model is trained using a least-squares objective to learn a single linear relationship between features and the target.

4. Evaluation metrics  
   Model performance is evaluated using mean squared error (MSE), root mean squared error (RMSE), and R² score.

## Failure Modes
Linear regression performs poorly when the relationship between features and the target is strongly non-linear, when outliers dominate the data, or when important interaction effects are present.

## Data Used In Notebook
The California Housing dataset, consisting of demographic and geographic features used to predict median house values.

## Comparison Notes (Optional)
Linear regression is highly interpretable and computationally efficient, but it lacks the flexibility of tree-based or neural network models on complex datasets.
