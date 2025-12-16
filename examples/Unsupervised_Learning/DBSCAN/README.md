# DBSCAN

## Overview
This notebook demonstrates the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm using the `rice_ml` implementation. DBSCAN groups points based on local point density and explicitly identifies outliers, making it well-suited for discovering arbitrarily shaped clusters without specifying the number of clusters in advance.

## Notebook Structure
- Synthetic dataset generation with clear cluster structure and noise
- Visualization of raw data
- Application of DBSCAN with different parameter settings
- Visualization of discovered clusters and noise points

## Algorithm Intuition
DBSCAN defines clusters as regions of high point density separated by regions of low density. Points are classified as:
- **Core points** (sufficient nearby neighbors),
- **Border points** (near a core point but not dense themselves),
- **Noise points** (outliers).

Two key parameters control behavior: `eps` (neighborhood radius) and `min_samples` (minimum neighbors to form a core point).

## Evaluation and Interpretation
Because DBSCAN is unsupervised, evaluation is qualitative. Results are assessed through:
- Visual inspection of cluster assignments
- Identification of noise points
- Sensitivity analysis over different `eps` and `min_samples` values

## Failure Modes
DBSCAN struggles when cluster densities vary significantly or when data is high-dimensional. Poor parameter choices can lead to all points being labeled as noise or merged into a single cluster.

## Data Used In Notebook
A synthetically generated 2D dataset containing multiple clusters and injected noise, designed to highlight DBSCAN’s ability to separate dense regions and detect outliers.

## Comparison Notes
Compared to K-Means, DBSCAN does not require specifying the number of clusters and can detect non-spherical clusters. However, it is more sensitive to parameter selection and does not scale as well to high-dimensional data.
