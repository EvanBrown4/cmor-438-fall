import numpy as np
import pandas as pd
from typing import Literal, Optional, Union, Tuple

from src.rice_ml.utilities import *
from src.rice_ml.utilities._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]

def _validate_1d_array_no_type(y: ArrayLike) -> np.ndarray:
    """
    Validate and clean a 1-dimensional array-like input into a NumPy array.

    Parameters
    ----------
    y : ArrayLike
        Input array-like object. Can be list, tuple, numpy array, pandas
        Series, or DataFrame column. Must represent a 1D sequence.

    Returns
    -------
    np.ndarray
        A clean NumPy array guaranteed to be 1-dimensional, non-empty,
        and containing finite values if numeric.

    Raises
    ------
    ValueError
        If the array is empty or has incorrect dimensionality.
    """
    ## Convert input to a NumPy array.
    y = np.asarray(y)

    ## Ensure the array is exactly 1-dimensional.
    if y.ndim != 1:
        raise ValueError(f"Expected 1D. Got {y.ndim} dimensions.")

    ## Validate non-empty input.
    if (y.size == 0):
        raise ValueError("Array cannot be empty.")
    
    ## Check that numeric values are finite.
    _check_finite_if_numeric(y)
    
    return y

def _pairwise_distances(XA: np.ndarray, XB: np.ndarray, dist_metric: str) -> np.ndarray:
    """
    Compute pairwise distances between rows of two matrices XA and XB.

    Parameters
    ----------
    XA : ndarray of shape (n_a, d)
        First input matrix.
    XB : ndarray of shape (n_b, d)
        Second input matrix.
    dist_metric : {"euclidean", "manhattan"}
        Distance metric to compute.

    Returns
    -------
    ndarray of shape (n_a, n_b)
        Pairwise distances between each row of XA and XB.

    Notes
    -----
    Uses fully vectorized NumPy operations with no Python loops.
    """
    ## Branch for Euclidean distance computations.
    if dist_metric == "euclidean":
        # compute squared norms
        aa = np.sum(XA * XA, axis=1, keepdims=True)       # (n_a, 1)
        bb = np.sum(XB * XB, axis=1, keepdims=True).T     # (1, n_b)

        # compute squared distances using identity
        D2 = np.maximum(aa + bb - 2.0 * XA @ XB.T, 0.0)

        # return sqrt of squared distances
        return np.sqrt(D2, dtype=float)

    ## Branch for Manhattan distances.
    elif dist_metric == "manhattan":
        # broadcast subtraction
        diff = XA[:, None, :] - XB[None, :, :]
        return np.sum(np.abs(diff), axis=2, dtype=float)

    ## Unsupported metric fallback.
    else:
        raise ValueError("Unsupported metric.")

def _neighbors(X_train: np.ndarray, X_query: np.ndarray, n_neighbors: int, dist_metric: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute nearest neighbors for each row in X_query relative to X_train.

    Parameters
    ----------
    X_train : ndarray of shape (n_train, d)
        Training samples.
    X_query : ndarray of shape (n_query, d)
        Query samples.
    n_neighbors : int
        Number of neighbors to compute.
    dist_metric : {"euclidean", "manhattan"}
        Distance metric.

    Returns
    -------
    distances : ndarray of shape (n_query, n_neighbors)
        Sorted distances of nearest neighbors.
    indices : ndarray of shape (n_query, n_neighbors)
        Corresponding indices in X_train.

    Raises
    ------
    ValueError
        If `n_neighbors` exceeds dataset size.
    """
    ## Compute pairwise distances.
    D = _pairwise_distances(X_query, X_train, dist_metric)

    ## Ensure n_neighbors is valid.
    if n_neighbors > X_train.shape[0]:
        raise ValueError(f"n_neighbors={n_neighbors} cannot exceed number of training samples={X_train.shape[0]}.")

    # use argpartition to get k smallest
    idx = np.argpartition(D, kth=n_neighbors - 1, axis=1)[:, :n_neighbors]

    # fully sort these k results
    row_indices = np.arange(D.shape[0])[:, None]
    dsel = D[row_indices, idx]
    order = np.argsort(dsel, axis=1)
    idx_sorted = idx[row_indices, order]
    d_sorted = dsel[row_indices, order]

    return d_sorted, idx_sorted

def _weights_from_distances(dist: np.ndarray, scheme: str, eps: float = 1e-12) -> np.ndarray:
    """
    Compute neighbor weights from distances.

    Parameters
    ----------
    dist : ndarray of shape (n_query, k)
        Distances to neighbors.
    scheme : {"uniform", "distance"}
        Weighting scheme to apply.
    eps : float, default=1e-12
        Minimum distance for stabilization of inverse-distance.

    Returns
    -------
    ndarray of shape (n_query, k)
        Non-normalized weights.

    """
    ## Uniform weighting: all ones.
    if scheme == "uniform":
        return np.ones_like(dist, dtype=float)

    ## Distance weighting: handle zero distances first.
    zero_mask = (dist <= eps)
    w = np.empty_like(dist, dtype=float)

    ## Handle rows with zero distances.
    any_zero = zero_mask.any(axis=1)
    if np.any(any_zero):
        w[any_zero] = zero_mask[any_zero].astype(float)

    ## Rows without zero distances → inverse-distance weighting.
    if np.any(~any_zero):
        w[~any_zero] = 1.0 / np.maximum(dist[~any_zero], eps)

    return w

class _KNNBase:
    """
    Base class for KNN classifier and regressor.

    Provides shared initialization, fit(), and neighbor search functionality.

    Attributes
    ----------
    n_neighbors : int
        Number of neighbors.
    distance : {"euclidean", "manhattan"}
        Distance metric.
    weights : {"uniform", "distance"}
        Weighting method.
    """
    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        distance: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        ## Validate constructor parameters.
        if distance != "euclidean" and distance != "manhattan":
            raise ValueError("Distance must be either euclidean or manhattan.")
        
        self.n_neighbors = int(n_neighbors)
        self.distance = distance
        self.weights = weights
        self._X = None
        self._y = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        """
        Fit the KNN model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        ## Validate inputs.
        X = _validate_2d_array(X)
        y = _validate_1d_array_no_type(y)
        _check_same_shape(X, y, "X", "y")

        ## Store training data.
        self._X = X
        self._y = y

        return self
    
    def _check_is_fitted(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Internal helper to verify that the model has been fitted.

        Returns
        -------
        (X_train, y_train) : tuple of ndarrays

        Raises
        ------
        RuntimeError
            If the model has not been fitted.
        """
        ## Ensure both X and y are set.
        if self._X is None or self._y is None:
            raise RuntimeError("Model is not fitted. Call fit(X, y) first.")
        return self._X, self._y
    
    def kneighbors(
            self,
            X: ArrayLike,
            n_neighbors: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute K nearest neighbors of each query point.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
            Query points.
        n_neighbors : int, optional
            Number of neighbors. Defaults to the estimator's configured
            `n_neighbors`.

        Returns
        -------
        distances : ndarray of shape (n_query, n_neighbors)
            Sorted neighbor distances.
        indices : ndarray of shape (n_query, n_neighbors)
            Corresponding indices in the training set.
        """
        ## Ensure model is fitted.
        X_train, _ = self._check_is_fitted()

        ## Validate query matrix.
        Xq = _validate_2d_array(X)

        ## Ensure feature dimensionality matches.
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(
                f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}."
            )

        ## Default to estimator setting if not provided.
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        return _neighbors(X_train, Xq, n_neighbors, self.distance)

class KNNClassifier(_KNNBase):
    """
    K-Nearest Neighbors classifier.

    Predicts classes based on weighted or unweighted neighbor voting.

    Attributes
    ----------
    classes_ : ndarray
        Sorted unique class labels seen during fitting.
    """
    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        distance: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, distance=distance, weights=weights)
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        """
        Fit classifier and infer class labels.

        Returns
        -------
        self : object
        """
        ## Validate inputs.
        X = _validate_2d_array(X)
        y = _validate_1d_array_no_type(y)
        _check_same_length(X, y, "X", "y")

        ## Store data.
        self._X = X
        self._y = y
        self.classes_ = np.unique(self._y)

        return self
    
    def predict_proba(self, X_test: ArrayLike) -> np.ndarray:
        """
        Compute class probability estimates for each query point.

        Parameters
        ----------
        X_test : array-like of shape (n_query, n_features)

        Returns
        -------
        ndarray of shape (n_query, n_classes)
            Row-normalized probability distributions.
        """
        ## Retrieve training data.
        X_train, y_train = self._check_is_fitted()

        ## Ensure classes are known.
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        Xq = _validate_2d_array(X_test)

        ## Validate number of features.
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        ## Compute neighbors.
        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.distance)
        w = _weights_from_distances(dist, self.weights)
        
        ## Prepare probability matrix.
        n_query = Xq.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_query, n_classes), dtype=float)
        
        ## Compute weighted class counts.
        for i in range(n_query):
            neigh_labels = y_train[idx[i]]
            class_ids = np.searchsorted(self.classes_, neigh_labels)
            counts = np.bincount(class_ids, weights=w[i], minlength=n_classes)
            total = counts.sum()

            # normalize
            if total == 0:
                proba[i] = 1.0 / n_classes
            else:
                proba[i] = counts / total

        return proba

    def predict(self, X_test):
        """
        Predict class labels for test samples.

        Returns
        -------
        ndarray of shape (n_query,)
            Predicted class labels.
        """
        proba = self.predict_proba(X_test)
        best = np.argmax(proba, axis=1)

        ## Ensure model was fitted.
        if self.classes_ is None:
            raise RuntimeError("Model is not fitted.")

        return self.classes_[best]
    
    def score(self, X_test: ArrayLike, y_true: ArrayLike) -> float:
        """
        Compute accuracy score.

        Parameters
        ----------
        X_test : array-like of shape (n_query, n_features)
        y_true : array-like of shape (n_query,)

        Returns
        -------
        float
            Classification accuracy.
        """
        _, _ = self._check_is_fitted()
        y_true = _validate_1d_array_no_type(y_true)
        y_pred = self.predict(X_test)

        _check_same_length(y_true, y_pred, "y_true", "y_pred")

        return float(np.mean(y_pred == y_true))

class KNNRegressor(_KNNBase):
    """
    K-Nearest Neighbors regressor.

    Predicts continuous values using weighted averages of neighbor targets.
    """
    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        distance: Literal["euclidean", "manhattan"] = "euclidean",
        weights: Literal["uniform", "distance"] = "uniform",
    ) -> None:
        super().__init__(n_neighbors=n_neighbors, distance=distance, weights=weights)

    def fit(self, X: ArrayLike, y: ArrayLike) -> "KNNRegressor":
        """
        Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Numeric regression targets.

        Returns
        -------
        self : object
        """
        ## Validate input data.
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)
        _check_same_length(X, y, "X", "y")

        ## Store values.
        self._X = X
        self._y = y.astype(float)

        return self
    
    def predict(self, X_test: ArrayLike) -> np.ndarray:
        """
        Predict regression values for query points.

        Parameters
        ----------
        X_test : array-like of shape (n_query, n_features)

        Returns
        -------
        ndarray of shape (n_query,)
            Predicted numeric values.
        """
        ## Retrieve stored training data.
        X_train, y_train = self._check_is_fitted()
        Xq = _validate_2d_array(X_test)

        ## Validate feature count.
        if Xq.shape[1] != X_train.shape[1]:
            raise ValueError(f"X has {Xq.shape[1]} features, expected {X_train.shape[1]}.")

        ## Compute distances and indices.
        dist, idx = _neighbors(X_train, Xq, self.n_neighbors, self.distance)

        ## Compute weights.
        w = _weights_from_distances(dist, self.weights)

        ## Gather target values.
        n_vals = y_train[idx]

        ## Standard weighted prediction.
        weighted_sum = np.sum(w * n_vals, axis=1)
        total_weight = np.sum(w, axis=1)

        preds = weighted_sum / np.where(total_weight == 0, 1.0, total_weight)

        ## Handle zero-weight rows.
        zero_mask = (total_weight == 0)
        if np.any(zero_mask):
            preds[zero_mask] = np.mean(n_vals[zero_mask], axis=1)
        
        return preds
    
    def score(self, X_test: ArrayLike, y_true: ArrayLike) -> float:
        """
        Compute R^2 (coefficient of determination).

        Parameters
        ----------
        X_test : array-like
        y_true : array-like

        Returns
        -------
        float
            R^2 score.
        """
        _, _ = self._check_is_fitted()
        
        y_true = _validate_1d_array(y_true)
        y_pred = self.predict(X_test)

        _check_same_length(y_true, y_pred, "y_true", "y_pred")

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1 - ss_res / ss_tot)