# Decision Trees (Classification)

## Overview
This notebook demonstrates decision tree classifiers implemented from scratch in `rice_ml`. Decision trees recursively split the feature space to create interpretable, rule-based models suitable for multi-class classification.

## Example Walkthrough
1. Dataset description  
    Explanation and minor exploration and visualization of the data.

2. Preprocessing steps  
    Images are flattened into feature vectors. No feature scaling is required due to the tree-based nature of the model.

3. Model training  
    A decision tree classifier is trained by recursively selecting feature thresholds that improve node purity.

4. Evaluation metrics  
    Classification accuracy is used to evaluate predictive performance on held-out data.

## Failure Modes
Decision trees tend to overfit when allowed to grow too deep and can be unstable, producing very different trees with small changes in the training data.

## Data Used In Notebook
A digit classification dataset (handwritten digits), where images are represented as numeric feature vectors and labels correspond to digit classes.

## Comparison Notes (Optional)
Decision trees are more interpretable than neural networks but generally underperform ensemble methods such as random forests on this task.
