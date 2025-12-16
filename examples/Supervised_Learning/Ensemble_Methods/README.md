# Ensemble Methods (Random Forest)

## Overview
This notebook demonstrates ensemble learning using Random Forests implemented in `rice_ml`. Random forests combine many decision trees trained on bootstrapped samples of the data and aggregate their predictions, leading to improved generalization and reduced variance compared to a single tree.

Both classification and regression variants are explored.

## Example Walkthrough
1. Dataset description  
   Two datasets are used: a handwritten digit classification dataset for the classifier and a housing dataset for the regressor. This section consists of an explanation and minor exploration and visualization of the data.

2. Preprocessing steps  
   For classification, digit images are flattened into feature vectors. No normalization is applied, as tree-based models are invariant to feature scaling. For regression, raw numerical features are used directly.

3. Model training  
   Random forest models are trained by fitting multiple decision trees with random feature subsets and bootstrapped samples. Hyperparameters such as the number of trees, maximum depth, and minimum leaf size are set explicitly for reproducibility.

4. Evaluation metrics  
   Classification performance is evaluated using accuracy and confusion matrix visualizations. Regression performance is evaluated using error-based metrics such as mean squared error.

## Failure Modes
Random forests can be computationally expensive to train, especially as the number of trees increases. While they reduce overfitting compared to single trees, they can still struggle with very high-dimensional data or insufficient training samples.

## Data Used In Notebook
- **Classification:** MNIST handwritten digit dataset (28×28 grayscale images, subsampled for runtime).
- **Regression:** California Housing dataset from scikit-learn, containing demographic and geographic features used to predict median house values.

## Comparison Notes (Optional)
Random forests typically outperform single decision trees by reducing variance, but they sacrifice interpretability. Compared to neural networks, they require less tuning and no feature scaling but may scale poorly to very large datasets.
