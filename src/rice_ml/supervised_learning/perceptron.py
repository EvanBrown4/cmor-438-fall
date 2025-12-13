import numpy as np
import pandas as pd
from typing import Optional, Union

from src.rice_ml.utilities._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class PerceptronClassifier:
    """
    Perceptron binary classifier.

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate.
    max_iter : int, default=1000
        Maximum number of epochs.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    shuffle : bool, default=True
        Whether to shuffle training data.
    tol : int or None, default=None
        Tolerance for stopping criterion.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_features + 1,)
        Weights assigned to features.
    n_features_in : int
        Number of features seen during fit.
    errors_ : list of int
        Number of misclassifications per epoch.
    n_iter_ : int
        Number of epochs run.
    """

    def __init__(
        self,
        lr: float = 0.01,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        shuffle: bool = True,
        tol: Optional[int] = None,
    ):
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.shuffle = shuffle
        self.tol = tol

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column."""
        return np.column_stack([np.ones(X.shape[0]), X])

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """Compute linear combination."""
        return X @ self.coef_

    def _activation(self, z: np.ndarray) -> np.ndarray:
        """Apply threshold activation."""
        return (z >= 0).astype(int)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "PerceptronClassifier":
        """
        Fit the perceptron model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels (0 or 1).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)
        _check_same_length(X, y, "X", "y")

        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("PerceptronClassifier supports binary labels {0,1} only.")

        if self.fit_intercept:
            X = self._add_intercept(X)

        self.n_features_in = X.shape[1]
        self.coef_ = np.zeros(self.n_features_in)

        self.errors_ = []
        n = X.shape[0]
        idx = np.arange(n)

        for epoch in range(self.max_iter):
            if self.shuffle:
                np.random.shuffle(idx)

            errors = 0

            for i in idx:
                z = X[i] @ self.coef_
                y_hat = 1 if z >= 0 else 0

                update = self.lr * (y[i] - y_hat)

                if update != 0:
                    self.coef_ += update * X[i]
                    errors += 1

            self.errors_.append(errors)

            if self.tol is not None and errors <= self.tol:
                break

        self.n_iter_ = epoch + 1
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels (0 or 1).
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fit yet.")

        X = _validate_2d_array(X)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Expected {self.n_features_in} features, got {X.shape[1]}"
            )

        z = self._net_input(X)
        return self._activation(z)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return mean accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Mean accuracy.
        """
        y = _validate_1d_array(y)
        preds = self.predict(X)
        return np.mean(preds == y)