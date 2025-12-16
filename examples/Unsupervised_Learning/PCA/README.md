# Principal Component Analysis (PCA)

## Overview
This notebook demonstrates Principal Component Analysis (PCA) using the from-scratch implementation in `rice_ml`. PCA is a dimensionality reduction technique that projects high-dimensional data onto a smaller set of orthogonal components that capture the maximum variance in the data.

The notebook emphasizes variance preservation and visualization as the primary motivations for PCA.

## Notebook Structure
- Loading and inspecting a high-dimensional dataset
- Fitting PCA
- Examining explained variance ratios for each component
- Visualizing cumulative explained variance
- Projecting data into a lower-dimensional space (2-D) for visualization

## Algorithm Intuition
PCA works by computing the eigenvectors (principal components) of the data’s covariance matrix. These components define directions of maximum variance, ordered from most to least informative.

By selecting only the top components, PCA reduces dimensionality while retaining as much information as possible.

## Evaluation and Interpretation
Because PCA is unsupervised, evaluation focuses on:
- Explained variance ratio per component
- Cumulative variance explained
- Visual separation of data after projection into 2D

The notebook highlights how choosing too few or too many components affects interpretability and information retention.

## Failure Modes
PCA performs poorly when:
- Important structure is non-linear
- Variance does not correspond to meaningful signal
- Features have vastly different scales and scaling is not appropriate for the task

In such cases, dimensionality reduction may discard important information.

## Data Used In Notebook
A high-dimensional numerical dataset where the first few features dominate the variance, making PCA’s variance-capturing behavior easy to observe.

## Comparison Notes
Unlike clustering algorithms, PCA does not group data points. Instead, it serves as a preprocessing or visualization tool and is often combined with classifiers or clustering methods downstream.
