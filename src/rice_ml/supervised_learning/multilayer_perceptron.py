import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional

from rice_ml.utilities._validation import (
    _validate_2d_array,
    _validate_1d_array,
    _check_same_length,
    _check_finite_if_numeric,
)

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class MLPClassifier:
    """
    Multi-Layer Perceptron (Neural Network) for binary classification.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(32,)
        The number of neurons in each hidden layer. For example, (32,) creates
        one hidden layer with 32 neurons, while (64, 32) creates two hidden
        layers with 64 and 32 neurons respectively.
    lr : float, default=0.01
        Learning rate for gradient descent. Controls the step size during
        weight updates.
    max_iter : int, default=1000
        Maximum number of training epochs (passes through the entire dataset).
    fit_intercept : bool, default=True
        Whether to include bias terms in the network. Currently always includes
        biases regardless of this parameter.
    random_state : int or None, default=None
        Random seed for weight initialization. Pass an int for reproducible
        results across multiple function calls.
    tol : float or None, default=None
        Tolerance for stopping criterion. Training stops when the change in
        loss is less than or equal to `tol`. If None, training runs for
        `max_iter` epochs.
    verbose : bool, default=False
        Whether to print loss information during training. Prints every 100
        epochs when True.

    Attributes
    ----------
    weights_ : list of ndarray
        Weight matrices for each layer. weights_[i] has shape
        (layer_sizes[i], layer_sizes[i+1]).
    biases_ : list of ndarray
        Bias vectors for each layer. biases_[i] has shape (layer_sizes[i+1],).
    losses_ : list of float
        Binary cross-entropy loss recorded at each epoch during training.
    n_iter_ : int
        Number of iterations (epochs) run during training.

    Examples
    --------
    >>> from src.rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
    >>> import numpy as np
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> y = np.array([0, 1, 1, 0])  # XOR problem
    >>> clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
    >>> clf.fit(X, y)
    >>> clf.predict([[0, 0]])
    array([0])

    Notes
    -----
    The MLP uses:
    - ReLU activation: f(x) = max(0, x) for hidden layers
    - Sigmoid activation: f(x) = 1 / (1 + exp(-x)) for output layer
    - Binary cross-entropy loss for training
    - Standard backpropagation for gradient computation
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

    def _init_weights(self, n_features: int):
        """
        Initialize weights and biases for all layers.

        Parameters
        ----------
        n_features : int
            Number of input features.

        Notes
        -----
        Weights are initialized with small random values from a normal
        distribution (mean=0, std=0.01). Biases are initialized to zero.
        """
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

    def _relu(self, Z):
        """
        Apply ReLU activation function.

        Parameters
        ----------
        Z : ndarray
            Pre-activation values.

        Returns
        -------
        ndarray
            Activated values (max(0, Z)).
        """
        return np.maximum(0, Z)

    def _relu_deriv(self, Z):
        """
        Compute derivative of ReLU activation.

        Parameters
        ----------
        Z : ndarray
            Pre-activation values.

        Returns
        -------
        ndarray
            Derivative values (1 if Z > 0, else 0).
        """
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        """
        Apply sigmoid activation function.

        Parameters
        ----------
        Z : ndarray
            Pre-activation values.

        Returns
        -------
        ndarray
            Activated values (1 / (1 + exp(-Z))).
        """
        return 1 / (1 + np.exp(-Z))

    def _forward(self, X):
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, 1)
            Network output (predicted probabilities).

        Notes
        -----
        Stores intermediate activations (As) and pre-activation values (Zs)
        for use in backpropagation.
        """
        A = X
        self.Zs = []
        self.As = [X]

        for W, b in zip(self.weights_[:-1], self.biases_[:-1]):
            Z = A @ W + b
            A = self._relu(Z)
            self.Zs.append(Z)
            self.As.append(A)

        # Output layer
        Z = A @ self.weights_[-1] + self.biases_[-1]
        A = self._sigmoid(Z)

        self.Zs.append(Z)
        self.As.append(A)

        return A

    def _loss(self, y, y_hat):
        """
        Compute binary cross-entropy loss.

        Parameters
        ----------
        y : ndarray of shape (n_samples, 1)
            True labels (0 or 1).
        y_hat : ndarray of shape (n_samples, 1)
            Predicted probabilities.

        Returns
        -------
        float
            Mean binary cross-entropy loss.

        Notes
        -----
        Adds small epsilon (1e-12) to prevent log(0) errors.
        """
        eps = 1e-12
        return -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

    def _backward(self, y):
        """
        Perform backpropagation and update weights.

        Parameters
        ----------
        y : ndarray of shape (n_samples, 1)
            True labels (0 or 1).

        Notes
        -----
        Computes gradients using the chain rule and updates all weights and
        biases using gradient descent.
        """
        m = len(y)
        grads_w = []
        grads_b = []

        # Output gradient
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

    def fit(self, X: ArrayLike, y: ArrayLike):
        """
        Train the multi-layer perceptron on the given data.

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

    def predict_proba(self, X: ArrayLike):
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Predicted probabilities for each class. First column is probability
            of class 0, second column is probability of class 1.
        """
        X = _validate_2d_array(X)
        y_hat = self._forward(X)
        return np.column_stack([1 - y_hat, y_hat])

    def predict(self, X: ArrayLike):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (0 or 1). Predictions are made using a
            threshold of 0.5 on the predicted probabilities.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def score(self, X: ArrayLike, y: ArrayLike):
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
        X = _validate_2d_array(X)
        preds = self.predict(X)
        y = _validate_1d_array(y)
        return np.mean(preds == y)