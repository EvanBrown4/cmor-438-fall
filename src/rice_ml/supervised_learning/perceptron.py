import numpy as np
from typing import Optional

from src.rice_ml.utilities._validation import *


class PerceptronClassifier:
    """
    Perceptron binary classifier.

    The perceptron is a linear classifier that learns a separating hyperplane
    for binary classification tasks. It updates weights based on misclassified
    samples using the perceptron learning rule. This implementation uses online
    learning (one sample at a time).

    Parameters
    ----------
    lr : float, default=0.01
        Learning rate (between 0.0 and 1.0). Controls the magnitude of weight
        updates.
    max_iter : int, default=1000
        Maximum number of passes over the training data (epochs).
    fit_intercept : bool, default=True
        Whether to calculate the intercept (bias term) for this model. If set
        to False, no intercept will be used in calculations.
    shuffle : bool, default=True
        Whether to shuffle training data before each epoch. Shuffling generally
        leads to better convergence.
    tol : int or None, default=None
        Tolerance for stopping criterion. If not None, training will stop when
        the number of misclassifications is less than or equal to `tol`. If None,
        training runs for `max_iter` epochs.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or (n_features + 1,)
        Weights assigned to the features. If `fit_intercept=True`, the first
        element is the bias term.
    n_features_in : int
        Number of features seen during fit (including intercept if used).
    errors_ : list of int
        Number of misclassifications in each epoch during training.
    n_iter_ : int
        Number of iterations (epochs) run during training.

    Examples
    --------
    >>> from src.rice_ml.supervised_learning.perceptron import PerceptronClassifier
    >>> import numpy as np
    >>> X = np.array([[0], [1], [2], [3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = PerceptronClassifier(max_iter=100)
    >>> clf.fit(X, y)
    >>> clf.predict([[1.5]])
    array([1])

    Notes
    -----
    The perceptron is guaranteed to converge only for linearly separable data.
    For non-separable data, it will continue updating weights until `max_iter`
    is reached or the tolerance criterion is met.
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
        """
        Append bias column of ones to the feature matrix.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples, n_features + 1)
            Feature matrix with prepended column of ones.
        """
        return np.column_stack([np.ones(X.shape[0]), X])

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Compute linear combination of inputs and weights.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Linear scores (net input values).
        """
        return X @ self.coef_

    def _activation(self, z: np.ndarray) -> np.ndarray:
        """
        Apply hard threshold activation function.

        Parameters
        ----------
        z : ndarray of shape (n_samples,)
            Linear scores (net input values).

        Returns
        -------
        ndarray of shape (n_samples,)
            Binary predictions (0 or 1).
        """
        return (z >= 0).astype(int)

    def fit(self, X, y):
        """
        Fit the perceptron model on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels. Must be binary (0 or 1).

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If y contains values other than 0 and 1.
        """
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)

        _check_same_length(X, y, "X", "y")
        _check_finite_if_numeric(X)
        _check_finite_if_numeric(y)

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

        # Training loop
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

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If X has a different number of features than seen during fit.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fit yet.")

        X = _validate_2d_array(X)
        _check_finite_if_numeric(X)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if X.shape[1] != self.n_features_in:
            raise ValueError(
                f"Expected {self.n_features_in} features, got {X.shape[1]}"
            )

        z = self._net_input(X)
        return self._activation(z)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.

        Returns
        -------
        float
            Mean accuracy of self.predict(X) with respect to y.
        """
        y = _validate_1d_array(y)
        preds = self.predict(X)
        return np.mean(preds == y)