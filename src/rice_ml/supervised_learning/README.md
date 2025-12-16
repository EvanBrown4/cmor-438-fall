# Supervised Learning

This module contains supervised learning algorithms implemented from scratch.

Each algorithm follows a consistent API:
- `fit(X, y)`
- `predict(X)`

## Algorithms Included

- Linear Regression
- Logistic Regression
- k-Nearest Neighbors
- Perceptron
- Multilayer Perceptron
- Decision Trees
- Regression Trees
- Ensemble Methods

## Design Notes

- Input validation is strict and centralized
- Models are deterministic when `random_state` is provided
- Focus is on algorithmic correctness and interpretability