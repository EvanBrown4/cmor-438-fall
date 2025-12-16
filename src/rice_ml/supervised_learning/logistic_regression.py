import numpy as np
import pandas as pd
import warnings
from typing import Union

from rice_ml.utilities._validation import (
    _validate_2d_array,
    _validate_1d_array,
    _check_same_length,
)

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class LogisticRegression:
    """
    Logistic Regression classifier using gradient descent.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    lr : float, default=0.1
        Learning rate.
    max_iter : int, default=1000
        Maximum iterations.
    tol : float, default=1e-3
        Tolerance for stopping.
    penalty : str, default="l2"
        Regularization penalty.
    C : float, default=1.0
        Inverse regularization strength.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features_in,)
        Estimated coefficients.
    n_features_in : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        lr: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-3,
        penalty: str = "l2",
        C: float = 1.0
    ):
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.C = C

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept column."""
        return np.c_[np.ones(X.shape[0]), X]

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function."""
        z = np.asarray(z, dtype=float)
        out = np.empty_like(z)

        pos = z >= 0
        neg = ~pos

        out[pos] = 1 / (1 + np.exp(-z[pos]))
        ez = np.exp(z[neg])
        out[neg] = ez / (1 + ez)

        return out

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (0 or 1).

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
            raise ValueError("LogisticRegression supports binary labels {0,1} only.")

        if self.fit_intercept:
            X = self._add_intercept(X)

        self.n_features_in = X.shape[1]
        self.coef_ = np.zeros(self.n_features_in)

        prev_loss = np.inf
        converged = False

        for _ in range(self.max_iter):
            z = np.dot(X, self.coef_)
            y_hat = self._sigmoid(z)

            grad = np.dot(X.T, (y_hat - y)) / len(y)

            if self.penalty == "l2":
                reg = self.coef_.copy()
                if self.fit_intercept:
                    reg[0] = 0
                grad += (1 / self.C) * reg

            self.coef_ -= self.lr * grad

            loss = -np.mean(y * np.log(y_hat + 1e-12) +
                            (1 - y) * np.log(1 - y_hat + 1e-12))

            if self.penalty == "l2":
                loss += (1 / (2 * self.C)) * np.sum(reg ** 2)

            if abs(prev_loss - loss) < self.tol:
                converged = True
                break
            prev_loss = loss
        
        if not converged and not np.all(np.isfinite(self.coef_)):
            warnings.warn("LogisticRegression did not converge", RuntimeWarning)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Compute probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probabilities for each class.
        """
        if not hasattr(self, "coef_"):
            raise RuntimeError("Model has not been fit yet.")

        X = _validate_2d_array(X)

        if self.fit_intercept:
            X = self._add_intercept(X)

        if X.shape[1] != self.n_features_in:
            raise ValueError("Input should have the same number of features as the fitted X.")

        probs = self._sigmoid(X @ self.coef_)
        return np.column_stack([1 - probs, probs])

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
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

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
        y_pred = self.predict(X)
        return np.mean(y_pred == y)