# K-Means Clustering

## Overview
This notebook demonstrates the K-Means clustering algorithm using the from-scratch implementation in `rice_ml`. K-Means is an unsupervised learning algorithm that partitions data into a fixed number of clusters by minimizing within-cluster variance.

The notebook emphasizes both the intuition behind the algorithm and the practical challenges of choosing an appropriate number of clusters.

## Notebook Structure
- Synthetic dataset generation with clear cluster structure
- Visualization of the raw data
- Application of K-Means with varying values of `k`
- Visualization of cluster assignments and centroids
- Discussion of good vs. poor choices of `k`

## Algorithm Intuition
K-Means alternates between:
1. Assigning each point to the nearest cluster centroid
2. Updating centroids as the mean of assigned points

This process repeats until convergence. The algorithm implicitly assumes clusters are spherical and roughly equal in size.

## Evaluation and Interpretation
Since K-Means is unsupervised, evaluation is primarily qualitative:
- Visual inspection of cluster separation
- Inspection of centroid placement
- Comparison of results across different values of `k`

The notebook also highlights how incorrect choices of `k` can lead to misleading cluster assignments.

## Failure Modes
K-Means performs poorly when clusters are non-spherical, overlapping, or vary significantly in density. It is also sensitive to initialization and requires the number of clusters to be specified in advance.

## Data Used In Notebook
A synthetically generated two-dimensional dataset designed to clearly illustrate clustering behavior and the impact of different values of `k`.

## Comparison Notes
Compared to DBSCAN, K-Means is simpler and faster but cannot detect noise or arbitrarily shaped clusters. Also, it requires specifying the number of clusters beforehand.
