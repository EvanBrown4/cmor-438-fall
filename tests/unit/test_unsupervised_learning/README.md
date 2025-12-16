# Unsupervised Learning Unit Tests

This directory contains **unit tests for unsupervised learning algorithms** in `rice_ml`.

These tests verify correctness without reliance on labels or downstream models.

---

## Algorithms Covered

- Principal Component Analysis (PCA)
- k-Means Clustering
- Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
- Community Detection

---

## What Is Tested

- Shape and type correctness
- Algorithm-specific invariants
- Deterministic behavior where applicable
- Proper handling of edge cases and degenerate inputs

---

## Design Principle

Each test file corresponds to **one algorithm** and verifies behavior
without relying on other components or pipelines.