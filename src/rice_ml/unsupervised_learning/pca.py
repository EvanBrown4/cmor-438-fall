import numpy as np
import pandas as pd
from typing import Optional, Union

from rice_ml.utilities._validation import _validate_2d_array

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class PrincipalComponentAnalysis:
    """
    Principal Component Analysis (PCA) using SVD.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of principal components to retain.
    whiten : bool, default=False
        Whether to scale components to unit variance.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.
    explained_variance_ : ndarray of shape (n_components,)
        Variance explained by each component.
    explained_variance_ratio_ : ndarray of shape (n_components,)
        Percentage of variance explained by each component.
    mean_ : ndarray of shape (n_features,)
        Mean of training data.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
    ):
        self.n_components = n_components
        self.whiten = whiten

    def fit(self, X: ArrayLike) -> "PrincipalComponentAnalysis":
        """
        Fit PCA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : PrincipalComponentAnalysis
            Fitted estimator.
        """
        X = _validate_2d_array(X)

        n_samples, n_features = X.shape

        if self.n_components is not None:
            if not 1 <= self.n_components <= min(n_samples, n_features):
                raise ValueError(
                    "n_components must be between 1 and min(n_samples, n_features)."
                )

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        if n_samples <= 1:
            explained_variance = np.zeros_like(S)
            total_variance = 0.0
        else:
            explained_variance = (S ** 2) / (n_samples - 1)
            total_variance = explained_variance.sum()


        k = self.n_components or Vt.shape[0]

        self.components_ = Vt[:k]
        self.explained_variance_ = explained_variance[:k]

        if total_variance == 0.0:
            self.explained_variance_ratio_ = np.zeros_like(self.explained_variance_)
        else:
            self.explained_variance_ratio_ = (
                self.explained_variance_ / total_variance
            )

        if self.whiten:
            self._scaling_ = np.sqrt(self.explained_variance_)
        else:
            self._scaling_ = None

        self.n_features_in_ = n_features

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        if not hasattr(self, "components_"):
            raise RuntimeError("PCA has not been fit yet.")

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}."
            )

        X_centered = X - self.mean_
        Z = X_centered @ self.components_.T

        if self.whiten:
            Z /= self._scaling_

        return Z

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        """
        Fit and transform data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: ArrayLike) -> np.ndarray:
        """
        Reconstruct data from PCA space.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Transformed data.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        if not hasattr(self, "components_"):
            raise RuntimeError("PCA has not been fit yet.")

        Z = _validate_2d_array(Z)

        if Z.shape[1] != self.components_.shape[0]:
            raise ValueError("Input has incorrect number of components.")

        if self.whiten:
            Z = Z * self._scaling_

        return Z @ self.components_ + self.mean_