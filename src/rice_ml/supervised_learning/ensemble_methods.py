import numpy as np
import pandas as pd
from typing import Optional, Literal, Union

from rice_ml.utilities._validation import _validate_2d_array, _validate_1d_array, _check_same_length
from rice_ml.utilities import r2_score
from rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class RandomForestClassifier:
    """
    Random Forest Classifier using CART-style decision trees.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at a leaf.
    max_features : {"sqrt", "log2", "all"} or int or None, default="sqrt"
        Number of features per split.
    bootstrap : bool, default=True
        Whether to use bootstrap samples.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        Fitted trees.
    _feature_indices_ : list of ndarray
        Feature indices per tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    _n_features_per_tree : int
        Number of features per tree.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Literal["sqrt", "log2", "all"] | int] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: list[DecisionTreeClassifier] = []
        self._feature_indices_: list[np.ndarray] = []
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self._n_features_per_tree: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestClassifier":
        """
        Build forest of trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)
        _check_same_length(X, y, "X", "y")

        self._validate_parameters()

        self.n_samples_, self.n_features_ = X.shape
        self._n_features_per_tree = self._resolve_max_features()

        self.trees_.clear()
        self._feature_indices_.clear()

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                X_train, y_train = self._bootstrap_sample(X, y, rng)
            else:
                X_train, y_train = X, y

            feat_idx = self._feature_subsample(self.n_features_, rng)
            X_sub = X_train[:, feat_idx]

            tree_seed = rng.integers(0, 2**32 - 1)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._n_features_per_tree,
                random_state=tree_seed,
            )

            tree.fit(X_sub, y_train)

            self.trees_.append(tree)
            self._feature_indices_.append(feat_idx)

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
            Predicted labels (majority vote).
        """
        if len(self.trees_) == 0:
            raise RuntimeError("RandomForestClassifier has not been fit yet.")

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self._feature_indices_)
        ])

        final = []
        for i in range(all_preds.shape[1]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            final.append(vals[np.argmax(counts)])

        return np.array(final, dtype=int)

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

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Create bootstrap sample."""
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)
        return X[idx], y[idx]

    def _feature_subsample(self, n_features: int, rng: np.random.Generator):
        """Select random feature subset."""
        k = self._n_features_per_tree
        if k > n_features:
            raise ValueError("Resolved max_features exceeds n_features.")
        return rng.choice(n_features, size=k, replace=False)

    def _validate_parameters(self):
        """Validate hyperparameters."""
        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")

        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth <= 0
        ):
            raise ValueError("max_depth must be a positive integer or None.")

        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")

        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1.")

        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features must be >= 1 if integer.")
        elif self.max_features not in ("sqrt", "log2", "all", None):
            raise ValueError(
                "max_features must be int, 'sqrt', 'log2', 'all', or None."
            )

        if not isinstance(self.bootstrap, bool):
            raise ValueError("bootstrap must be a boolean.")

    def _resolve_max_features(self) -> int:
        """Convert max_features to integer."""
        n = self.n_features_
        mf = self.max_features

        if mf is None or mf == "all":
            return n

        if mf == "sqrt":
            return max(1, int(np.sqrt(n)))

        if mf == "log2":
            return max(1, int(np.log2(n)))

        if isinstance(mf, int):
            if mf < 1 or mf > n:
                raise ValueError("max_features must be in [1, n_features].")
            return mf

        raise ValueError("Invalid max_features value.")


class RandomForestRegressor:
    """
    Random Forest Regressor.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees.
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at a leaf.
    max_features : {"sqrt", "log2", "all"} or int or None, default="sqrt"
        Number of features per split.
    bootstrap : bool, default=True
        Whether to use bootstrap samples.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    trees_ : list of DecisionTreeRegressor
        Fitted trees.
    _feature_indices_ : list of ndarray
        Feature indices per tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    _n_features_per_tree : int
        Number of features per tree.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Literal["sqrt", "log2", "all"] | int] = "sqrt",
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.trees_: list[DecisionTreeRegressor] = []
        self._feature_indices_: list[np.ndarray] = []
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self._n_features_per_tree: int = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "RandomForestRegressor":
        """
        Build forest of trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)
        _check_same_length(X, y, "X", "y")

        self._validate_parameters()

        self.n_samples_, self.n_features_ = X.shape
        self._n_features_per_tree = self._resolve_max_features()

        self.trees_.clear()
        self._feature_indices_.clear()

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                X_train, y_train = self._bootstrap_sample(X, y, rng)
            else:
                X_train, y_train = X, y

            feat_idx = self._feature_subsample(self.n_features_, rng)
            X_sub = X_train[:, feat_idx]

            tree_seed = rng.integers(0, 2**32 - 1)

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._n_features_per_tree,
                random_state=tree_seed,
            )

            tree.fit(X_sub, y_train)

            self.trees_.append(tree)
            self._feature_indices_.append(feat_idx)

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict regression values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values (average).
        """
        if len(self.trees_) == 0:
            raise RuntimeError("RandomForestRegressor has not been fit yet.")

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self._feature_indices_)
        ])

        return np.mean(all_preds, axis=0)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return R^2 score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values.

        Returns
        -------
        float
            R^2 score.
        """
        y = _validate_1d_array(y)
        y_pred = self.predict(X)

        return r2_score(y, y_pred)

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
        """Create bootstrap sample."""
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)
        return X[idx], y[idx]

    def _feature_subsample(self, n_features: int, rng: np.random.Generator):
        """Select random feature subset."""
        k = self._n_features_per_tree
        if k > n_features:
            raise ValueError("Resolved max_features exceeds n_features.")
        return rng.choice(n_features, size=k, replace=False)

    def _validate_parameters(self):
        """Validate hyperparameters."""
        if not isinstance(self.n_estimators, int) or self.n_estimators <= 0:
            raise ValueError("n_estimators must be a positive integer.")

        if self.max_depth is not None and (
            not isinstance(self.max_depth, int) or self.max_depth <= 0
        ):
            raise ValueError("max_depth must be a positive integer or None.")

        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")

        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1.")

        if isinstance(self.max_features, int):
            if self.max_features < 1:
                raise ValueError("max_features must be >= 1 if integer.")
        elif self.max_features not in ("sqrt", "log2", "all", None):
            raise ValueError(
                "max_features must be int, 'sqrt', 'log2', 'all', or None."
            )

        if not isinstance(self.bootstrap, bool):
            raise ValueError("bootstrap must be a boolean.")

    def _resolve_max_features(self) -> int:
        """Convert max_features to integer."""
        n = self.n_features_
        mf = self.max_features

        if mf is None or mf == "all":
            return n
        if mf == "sqrt":
            return max(1, int(np.sqrt(n)))
        if mf == "log2":
            return max(1, int(np.log2(n)))
        if isinstance(mf, int):
            if mf < 1 or mf > n:
                raise ValueError("max_features must be in [1, n_features].")
            return mf

        raise ValueError("Invalid max_features value.")