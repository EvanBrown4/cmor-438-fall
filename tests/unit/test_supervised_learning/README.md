# Supervised Learning Unit Tests

This directory contains **unit tests for supervised learning algorithms** in `rice_ml`.

Each test file validates the correctness of a single algorithm in isolation.

---

## Algorithms Covered

- k-Nearest Neighbors (KNN)
- Linear Regression
- Logistic Regression
- Perceptron
- Multilayer Perceptron (MLP)
- Decision Trees
- Regression Trees
- Ensemble Method (Random Forest)

---

## What Is Tested

- Input validation (shapes, types, parameter bounds)
- Output shapes and basic invariants
- Core algorithm logic on small, deterministic datasets
- Correct handling of edge cases

---

## Design Principle

Each test file corresponds to **one algorithm** and verifies behavior
without relying on other components or pipelines.