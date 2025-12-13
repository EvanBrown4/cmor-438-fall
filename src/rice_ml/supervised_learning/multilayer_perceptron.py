import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

from src.rice_ml.utilities._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class MLPClassifier:
    """
    Multi-Layer Perceptron for binary classification.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(32,)
        Number of neurons in each hidden layer.
    lr : float, default=0.01
        Learning rate.
    max_iter : int, default=1000
        Maximum number of epochs.
    fit_intercept : bool, default=True
        Whether to include bias terms.
    random_state : int or None, default=None
        Random seed for weight initialization.
    tol : float or None, default=None
        Tolerance for stopping criterion.
    verbose : bool, default=False
        Whether to print loss information.

    Attributes
    ----------
    weights_ : list of ndarray
        Weight matrices for each layer.
    biases_ : list of ndarray
        Bias vectors for each layer.
    losses_ : list of float
        Binary cross-entropy loss per epoch.
    n_iter_ : int
        Number of epochs run.
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (32,),
        lr: float = 0.01,
        max_iter: int = 1000,
        fit_intercept: bool = True,
        random_state: Optional[int] = None,
        tol: Optional[float] = None,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.tol = tol
        self.verbose = verbose

    def _init_weights(self, n_features: int) -> None:
        """Initialize weights and biases."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        layer_sizes = (n_features,) + self.hidden_layer_sizes + (1,)

        self.weights_ = []
        self.biases_ = []

        for i in range(len(layer_sizes) - 1):
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros(layer_sizes[i+1])
            self.weights_.append(W)
            self.biases_.append(b)

    def _relu(self, Z: np.ndarray) -> np.ndarray:
        """Apply ReLU activation."""
        return np.maximum(0, Z)

    def _relu_deriv(self, Z: np.ndarray) -> np.ndarray:
        """Compute ReLU derivative."""
        return (Z > 0).astype(float)

    def _sigmoid(self, Z: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation."""
        return 1 / (1 + np.exp(-Z))

    def _forward(self, X: np.ndarray) -> np.ndarray:
        """Perform forward propagation."""
        A = X
        self.Zs = []
        self.As = [X]

        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            Z = A @ W + b
            A = self._relu(Z)
            self.Zs.append(Z)
            self.As.append(A)

        Z = A @ self.weights_[-1] + self.biases_[-1]
        A = self._sigmoid(Z)

        self.Zs.append(Z)
        self.As.append(A)

        return A

    def _loss(self, y: np.ndarray, y_hat: np.ndarray) -> float:
        """Compute binary cross-entropy loss."""
        eps = 1e-12
        return -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

    def _backward(self, y: np.ndarray) -> None:
        """Perform backpropagation and update weights."""
        m = len(y)
        grads_w = []
        grads_b = []

        dA = self.As[-1] - y.reshape(-1, 1)

        for i in reversed(range(len(self.weights_))):
            dW = self.As[i].T @ dA / m
            db = dA.mean(axis=0)

            grads_w.insert(0, dW)
            grads_b.insert(0, db)

            if i != 0:
                dZ = dA @ self.weights_[i].T
                dA = dZ * self._relu_deriv(self.Zs[i-1])

        for i in range(len(self.weights_)):
            self.weights_[i] -= self.lr * grads_w[i]
            self.biases_[i] -= self.lr * grads_b[i]

    def fit(self, X: ArrayLike, y: ArrayLike) -> "MLPClassifier":
        """
        Train the multi-layer perceptron.

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

        y = y.reshape(-1, 1)

        unique = np.unique(y)
        if not np.all(np.isin(unique, [0, 1])):
            raise ValueError("MLPClassifier supports binary labels {0,1} only.")

        self._init_weights(X.shape[1])
        self.losses_ = []

        prev_loss = np.inf

        for epoch in range(self.max_iter):
            y_hat = self._forward(X)
            loss = self._loss(y, y_hat)
            self.losses_.append(loss)

            self._backward(y)

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch} — Loss: {loss:.6f}")

            if self.tol is not None and abs(prev_loss - loss) <= self.tol:
                break

            prev_loss = loss

        self.n_iter_ = epoch + 1
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Probabilities for each class.
        """
        X = _validate_2d_array(X)
        y_hat = self._forward(X)
        return np.column_stack([1 - y_hat, y_hat])

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
        preds = self.predict(X)
        return np.mean(preds == y)