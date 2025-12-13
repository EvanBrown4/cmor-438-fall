import numpy as np
import pandas as pd
from collections import deque
from typing import Literal, Union

from src.rice_ml.utilities._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class DBSCAN:
    """
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    eps : float
        Maximum distance for two samples to be neighbors.
    min_samples : int, default=5
        Minimum samples in a neighborhood for a core point.
    metric : {"euclidean", "manhattan"}, default="euclidean"
        Distance metric.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels. Noise points are labeled -1.
    n_clusters_ : int
        Number of clusters found (excluding noise).
    core_samples_ : ndarray of shape (n_core_samples,)
        Indices of core samples.
    X_ : ndarray of shape (n_samples, n_features)
        Training data.
    """

    def __init__(
        self,
        eps: float,
        min_samples: int = 5,
        metric: Literal["euclidean", "manhattan"] = "euclidean",
    ) -> None:
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit(self, X: ArrayLike) -> "DBSCAN":
        """
        Perform DBSCAN clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : DBSCAN
            Fitted estimator.
        """
        X = _validate_2d_array(X)

        if self.eps <= 0:
            raise ValueError("eps must be positive.")

        if self.min_samples < 1:
            raise ValueError("min_samples must be >= 1.")

        if self.metric not in ["euclidean", "manhattan"]:
            raise ValueError("metric must be 'euclidean' or 'manhattan'.")

        self.X_ = X
        n_samples = X.shape[0]

        labels = np.full(n_samples, -1, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        core_mask = np.zeros(n_samples, dtype=bool)

        cluster_id = 0
        D = self._pairwise_distances(X)

        for i in range(n_samples):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = self._region_query(i, D)

            if len(neighbors) < self.min_samples:
                continue

            labels[i] = cluster_id
            core_mask[i] = True

            self._expand_cluster(
                i, neighbors, labels, visited, core_mask, D, cluster_id
            )

            cluster_id += 1

        self.labels_ = labels
        self.n_clusters_ = cluster_id
        self.core_samples_ = np.where(core_mask)[0]

        return self

    def fit_predict(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and return cluster labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        """
        self.fit(X)
        return self.labels_

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n_samples = X.shape[0]
        D = np.zeros((n_samples, n_samples))
        
        if self.metric == "euclidean":
            diff = X[:, None, :] - X[None, :, :]
            D = np.sqrt(np.sum(diff**2, axis=2))
        elif self.metric == "manhattan":
            diff = X[:, None, :] - X[None, :, :]
            D = np.sum(np.abs(diff), axis=2)
        
        return D

    def _region_query(self, i: int, D: np.ndarray) -> np.ndarray:
        """Find neighbors within eps radius."""
        return np.where(D[i] <= self.eps)[0]

    def _expand_cluster(
        self,
        i: int,
        neighbors: np.ndarray,
        labels: np.ndarray,
        visited: np.ndarray,
        core_mask: np.ndarray,
        D: np.ndarray,
        cluster_id: int,
    ) -> None:
        """Expand cluster using breadth-first search."""
        queue = deque(neighbors)

        while queue:
            j = queue.popleft()

            if not visited[j]:
                visited[j] = True
                j_neighbors = self._region_query(j, D)

                if len(j_neighbors) >= self.min_samples:
                    core_mask[j] = True
                    queue.extend(j_neighbors)

            if labels[j] == -1:
                labels[j] = cluster_id