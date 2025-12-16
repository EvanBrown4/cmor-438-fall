# Regression Trees

## Overview
This notebook demonstrates regression trees implemented from scratch in `rice_ml`. Regression trees model continuous targets by recursively partitioning the feature space and predicting the mean value within each leaf node.

## Example Walkthrough
1. Dataset description  
   Explanation and minor exploration of the data.

2. Preprocessing steps  
   Minimal preprocessing is required. Features are used directly without normalization, as tree-based models are invariant to feature scaling.

3. Model training  
   The regression tree is trained by recursively selecting feature thresholds that minimize variance (squared error) within each split. Hyperparameters such as maximum depth and minimum samples per leaf are set explicitly.

4. Evaluation metrics  
   Model performance is evaluated using mean squared error (MSE) and root mean squared error (RMSE).

## Failure Modes
Regression trees are prone to overfitting when allowed to grow deep and can produce piecewise-constant predictions that fail to capture smooth trends in the data.

## Data Used In Notebook
The California Housing dataset, consisting of demographic and geographic features used to predict median house values.

## Comparison Notes (Optional)
Regression trees can model non-linear relationships unlike linear regression, but individual trees typically underperform ensemble methods such as random forests.
