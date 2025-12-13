import numpy as np
import pandas as pd
from typing import Optional, Literal, Union

from src.rice_ml.utilities._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class KMeans:
    """
    K-Means clustering algorithm.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    init : {"k-means++", "random"}, default="k-means++"
        Initialization method.
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance for convergence.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centroids.
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each sample.
    inertia_ : float
        Sum of squared distances to closest centroid.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_clusters: int,
        init: Literal["k-means++", "random"] = "k-means++",
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: ArrayLike) -> "KMeans":
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : KMeans
            Fitted estimator.
        """
        X = _validate_2d_array(X)

        n_samples, n_features = X.shape

        if n_samples == 0:
            raise ValueError("Input X must contain at least one sample.")

        if not isinstance(self.n_clusters, int) or self.n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")

        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of samples.")

        if self.init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++' or 'random'.")

        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")

        if self.tol < 0:
            raise ValueError("tol must be non-negative.")

        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = self._initialize_centers(X, rng)

        for i in range(self.max_iter):
            labels = self._assign_labels(X, self.cluster_centers_)
            new_centers = self._compute_centers(X, labels, rng)

            shift = np.linalg.norm(self.cluster_centers_ - new_centers)
            self.cluster_centers_ = new_centers

            if shift <= self.tol:
                break

        self.labels_ = self._assign_labels(X, self.cluster_centers_)
        self.inertia_ = self._compute_inertia(X, self.cluster_centers_, self.labels_)
        self.n_iter_ = i + 1

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Model has not been fit yet.")

        X = _validate_2d_array(X)
        return self._assign_labels(X, self.cluster_centers_)

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform to cluster-distance space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            Distances to each cluster center.
        """
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Model has not been fit yet.")

        X = _validate_2d_array(X)
        return self._pairwise_distances(X, self.cluster_centers_)

    def _initialize_centers(
        self, 
        X: np.ndarray, 
        rng: np.random.Generator
    ) -> np.ndarray:
        """Initialize cluster centers."""
        if self.init == "random":
            indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
            return X[indices].copy()

        centers = []
        indices = rng.choice(X.shape[0])
        centers.append(X[indices])

        for _ in range(1, self.n_clusters):
            D = self._pairwise_distances(X, np.array(centers))
            dist_sq = np.min(D, axis=1) ** 2
            probs = dist_sq / np.sum(dist_sq)
            index = rng.choice(X.shape[0], p=probs)
            centers.append(X[index])

        return np.array(centers)

    def _assign_labels(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign samples to nearest center."""
        D = self._pairwise_distances(X, centers)
        return np.argmin(D, axis=1)

    def _compute_centers(
        self, 
        X: np.ndarray, 
        labels: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """Compute new cluster centers."""
        centers = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            members = X[labels == k]

            if len(members) == 0:
                centers[k] = X[rng.integers(0, X.shape[0])]
            else:
                centers[k] = members.mean(axis=0)

        return centers

    def _compute_inertia(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute inertia (within-cluster sum of squares)."""
        dist = self._pairwise_distances(X, centers)
        return float(np.sum((dist[np.arange(len(labels)), labels]) ** 2))

    def _pairwise_distances(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        diff = A[:, None, :] - B[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))