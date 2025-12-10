import numpy as np
from typing import Optional, Literal

from src.rice_ml.utilities import *
from src.rice_ml.utilities._validation import *

from src.rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)


class RandomForestClassifier:
    """
    Random Forest Classifier using CART-style decision trees.
    
    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree. If None, nodes are expanded until all leaves
        are pure or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2', 'all'} or int or None, default='sqrt'
        Number of features to consider when looking for the best split:
        - If 'sqrt', then max_features=sqrt(n_features).
        - If 'log2', then max_features=log2(n_features).
        - If 'all' or None, then max_features=n_features.
        - If int, then consider max_features features at each split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
    random_state : int or None, default=None
        Controls the randomness of the bootstrapping and feature sampling.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    trees_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.
    _feature_indices_ : list of ndarray
        The feature indices selected for each tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    _n_features_per_tree : int
        Number of features used per tree (resolved from max_features).

    Examples
    --------
    >>> from src.rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> clf.fit(X, y)
    >>> clf.predict([[1.5, 1.5]])
    array([1])
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

        # Will be set during fit
        self.trees_: list[DecisionTreeClassifier] = []
        self._feature_indices_: list[np.ndarray] = []
        self.n_samples_: int = 0
        self.n_features_: int = 0
        self._n_features_per_tree: int = 0

    def fit(self, X, y):
        """
        Build a forest of trees from the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)

        _check_same_length(X, y, "X", "y")
        _check_finite_if_numeric(X)
        _check_finite_if_numeric(y)

        self._validate_parameters()

        self.n_samples_, self.n_features_ = X.shape
        self._n_features_per_tree = self._resolve_max_features()

        self.trees_.clear()
        self._feature_indices_.clear()

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            # Bootstrap sampling over rows
            if self.bootstrap:
                X_train, y_train = self._bootstrap_sample(X, y, rng)
            else:
                X_train, y_train = X, y

            # Feature subsampling (true RF)
            feat_idx = self._feature_subsample(self.n_features_, rng)
            X_sub = X_train[:, feat_idx]

            # Train a fresh decision tree on this sample
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._n_features_per_tree,
            )
            tree.fit(X_sub, y_train)

            self.trees_.append(tree)
            self._feature_indices_.append(feat_idx)

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
            Predicted class labels (majority vote from all trees).
        """
        if len(self.trees_) == 0:
            raise RuntimeError("RandomForestClassifier has not been fit yet.")

        X = _validate_2d_array(X)
        _check_finite_if_numeric(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        # Collect predictions from all trees: shape (n_estimators, n_samples)
        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self._feature_indices_)
        ])

        # Majority vote per sample
        final = []
        for i in range(all_preds.shape[1]):
            vals, counts = np.unique(all_preds[:, i], return_counts=True)
            final.append(vals[np.argmax(counts)])

        return np.array(final)

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

    def _bootstrap_sample(self, X, y, rng: np.random.Generator):
        """
        Create a bootstrap sample from the dataset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        tuple of (ndarray, ndarray)
            Bootstrap sample of X and y (sampled with replacement).
        """
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)
        return X[idx], y[idx]

    def _feature_subsample(self, n_features: int, rng: np.random.Generator):
        """
        Select a random subset of features for a tree.

        Parameters
        ----------
        n_features : int
            Total number of features available.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ndarray
            Indices of selected features (sampled without replacement).
        """
        k = self._n_features_per_tree
        if k > n_features:
            raise ValueError("Resolved max_features exceeds n_features.")
        return rng.choice(n_features, size=k, replace=False)

    def _validate_parameters(self):
        """
        Validate hyperparameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
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
        """
        Convert max_features parameter into an integer feature count.

        Returns
        -------
        int
            Number of features to use per tree.

        Raises
        ------
        ValueError
            If max_features is invalid or out of range.
        """
        n = self.n_features_
        mf = self.max_features

        # Use all features
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

    def __repr__(self):
        return (
            f"RandomForestClassifier("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"max_features={self.max_features})"
        )


class RandomForestRegressor:
    """
    Random Forest Regressor using CART-style decision trees.

    A random forest is a meta estimator that fits a number of decision tree
    regressors on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting. The sub-sample
    size is controlled with the bootstrap parameter. Each tree is trained on
    a bootstrap sample of the data and a random subset of features.

    Parameters
    ----------
    n_estimators : int, default=100
        The number of trees in the forest.
    max_depth : int or None, default=None
        Maximum depth of each tree. If None, nodes are expanded until all leaves
        contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : {'sqrt', 'log2', 'all'} or int or None, default='sqrt'
        Number of features to consider when looking for the best split:
        - If 'sqrt', then max_features=sqrt(n_features).
        - If 'log2', then max_features=log2(n_features).
        - If 'all' or None, then max_features=n_features.
        - If int, then consider max_features features at each split.
    bootstrap : bool, default=True
        Whether bootstrap samples are used when building trees. If False, the
        whole dataset is used to build each tree.
    random_state : int or None, default=None
        Controls the randomness of the bootstrapping and feature sampling.
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    trees_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    _feature_indices_ : list of ndarray
        The feature indices selected for each tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    _n_features_per_tree : int
        Number of features used per tree (resolved from max_features).

    Examples
    --------
    >>> from src.rice_ml.supervised_learning.ensemble_methods import RandomForestRegressor
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> reg = RandomForestRegressor(n_estimators=10, random_state=42)
    >>> reg.fit(X, y)
    >>> reg.predict([[1.5, 1.5]])
    array([1.5])
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

    def fit(self, X, y):
        """
        Build a forest of trees from the training set.

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
        _check_finite_if_numeric(X)
        _check_finite_if_numeric(y)

        self._validate_parameters()

        self.n_samples_, self.n_features_ = X.shape
        self._n_features_per_tree = self._resolve_max_features()

        self.trees_.clear()
        self._feature_indices_.clear()

        rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            if self.bootstrap:
                X_train, y_train = self._bootstrap_sample(X, y, rng)
            else:
                X_train, y_train = X, y

            # Feature subsampling
            feat_idx = self._feature_subsample(self.n_features_, rng)
            X_sub = X_train[:, feat_idx]

            # Train tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self._n_features_per_tree,
            )
            tree.fit(X_sub, y_train)

            self.trees_.append(tree)
            self._feature_indices_.append(feat_idx)

        return self

    def predict(self, X):
        """
        Predict regression values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values (average of predictions from all trees).
        """
        if len(self.trees_) == 0:
            raise RuntimeError("RandomForestRegressor has not been fit yet.")

        X = _validate_2d_array(X)
        _check_finite_if_numeric(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees_, self._feature_indices_)
        ])

        return np.mean(all_preds, axis=0)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        float
            R^2 score of self.predict(X) with respect to y.
        """
        y = _validate_1d_array(y)
        preds = self.predict(X)

        ss_res = np.sum((y - preds) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot

    def _bootstrap_sample(self, X, y, rng: np.random.Generator):
        """
        Create a bootstrap sample from the dataset.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        tuple of (ndarray, ndarray)
            Bootstrap sample of X and y (sampled with replacement).
        """
        n = X.shape[0]
        idx = rng.integers(0, n, size=n)
        return X[idx], y[idx]

    def _feature_subsample(self, n_features: int, rng: np.random.Generator):
        """
        Select a random subset of features for a tree.

        Parameters
        ----------
        n_features : int
            Total number of features available.
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        ndarray
            Indices of selected features (sampled without replacement).
        """
        k = self._n_features_per_tree
        if k > n_features:
            raise ValueError("Resolved max_features exceeds n_features.")
        return rng.choice(n_features, size=k, replace=False)

    def _validate_parameters(self):
        """
        Validate hyperparameters.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
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
        """
        Convert max_features parameter into an integer feature count.

        Returns
        -------
        int
            Number of features to use per tree.

        Raises
        ------
        ValueError
            If max_features is invalid or out of range.
        """
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

    def __repr__(self):
        return (
            f"RandomForestRegressor("
            f"n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"max_features={self.max_features})"
        )