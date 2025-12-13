import numpy as np
import pandas as pd
from typing import Optional, Literal, Union

from src.rice_ml.utilities import *
from src.rice_ml.utilities._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class _TreeNode:
    """Internal node structure for decision tree."""
    def __init__(
        self,
        feature: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
        value: Optional[float] = None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class _BaseDecisionTree:
    """
    Base class for decision tree implementations.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at a leaf.
    max_features : int or None, default=None
        Number of features per split.

    Attributes
    ----------
    root_ : _TreeNode
        Root node of fitted tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

        self.root_ = None
        self.n_samples_ = 0
        self.n_features_ = 0

    def fit(self, X: ArrayLike, y: ArrayLike) -> "_BaseDecisionTree":
        """
        Build decision tree.

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

        self.n_samples_, self.n_features_ = X.shape

        root = self._build_tree(X, y, depth=0)

        if root is None:
            raise RuntimeError("Tree failed to build: _build_tree returned None.")

        self.root_ = root
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict class labels or values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("DecisionTree model has not been fit yet.")

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        return np.array([self._traverse_tree(x, self.root_) for x in X])

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return score of predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels or values.

        Returns
        -------
        float
            Mean accuracy (classifier) or R^2 (regressor).
        """
        y = _validate_1d_array(y)
        preds = self.predict(X)

        if isinstance(self, DecisionTreeClassifier):
            return np.mean(preds == y)

        elif isinstance(self, DecisionTreeRegressor):
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)

            if ss_tot == 0:
                return 0.0

            return 1 - ss_res / ss_tot

        else:
            raise TypeError("Unknown tree type.")

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        """Recursively build tree."""
        n_samples, n_features = X.shape

        if len(np.unique(y)) == 1:
            return _TreeNode(value=self._leaf_value(y))

        if self.max_depth is not None and depth >= self.max_depth:
            return _TreeNode(value=self._leaf_value(y))

        if n_samples < self.min_samples_split:
            return _TreeNode(value=self._leaf_value(y))

        best = self._best_split(X, y)

        if best is None:
            return _TreeNode(value=self._leaf_value(y))

        feature, threshold = best

        X_left, y_left, X_right, y_right = self._split(X, y, feature, threshold)

        if len(y_left) == 0 or len(y_right) == 0:
            return _TreeNode(value=self._leaf_value(y))

        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)

        return _TreeNode(
            feature=feature,
            threshold=threshold,
            left=left,
            right=right,
            value=None,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find best split."""
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_split = None

        parent_impurity = self._impurity(y)

        for feature in range(n_features):
            values = np.unique(X[:, feature])

            if len(values) <= 1:
                continue

            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                X_l, y_l, X_r, y_r = self._split(X, y, feature, threshold)

                if len(y_l) < self.min_samples_leaf or len(y_r) < self.min_samples_leaf:
                    continue

                w_l = len(y_l) / n_samples
                w_r = len(y_r) / n_samples

                gain = parent_impurity - (
                    w_l * self._impurity(y_l) + w_r * self._impurity(y_r)
                )

                if gain > best_gain and gain > 0:
                    best_gain = gain
                    best_split = (feature, threshold)

        return best_split

    def _split(self, X: np.ndarray, y: np.ndarray, feature: int, threshold: float):
        """Split data."""
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return (
            X[left_mask], y[left_mask],
            X[right_mask], y[right_mask],
        )

    def _traverse_tree(self, x: np.ndarray, node: _TreeNode) -> float:
        """Traverse tree for prediction."""
        if node is None:
            raise RuntimeError("Invalid tree structure: reached a None node.")

        if node.value is not None:
            return node.value

        if node.feature is None or node.threshold is None:
            raise RuntimeError("Invalid tree node: missing feature or threshold.")

        if x[node.feature] <= node.threshold:
            next_node = node.left
        else:
            next_node = node.right

        if next_node is None:
            raise RuntimeError("Invalid tree structure: child node is None.")

        return self._traverse_tree(x, next_node)

    def get_depth(self) -> int:
        """Get tree depth."""
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("DecisionTree model has not been fit yet.")
        return self._get_depth(self.root_)

    def _get_depth(self, node: _TreeNode) -> int:
        """Recursively compute depth."""
        if node is None:
            return 0

        if node.value is not None:
            return 1

        left_depth = self._get_depth(node.left) if node.left is not None else 0
        right_depth = self._get_depth(node.right) if node.right is not None else 0

        return 1 + max(left_depth, right_depth)

    def get_n_leaves(self) -> int:
        """Get number of leaves."""
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("DecisionTree model has not been fit yet.")
        return self._get_n_leaves(self.root_)

    def _get_n_leaves(self, node: _TreeNode) -> int:
        """Recursively count leaves."""
        if node is None:
            return 0

        if node.value is not None:
            return 1
        
        left_leaves = self._get_n_leaves(node.left) if node.left is not None else 0
        right_leaves = self._get_n_leaves(node.right) if node.right is not None else 0

        return left_leaves + right_leaves

    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity."""
        raise NotImplementedError("_impurity must be implemented in subclasses.")

    def _leaf_value(self, y: np.ndarray) -> float:
        """Determine leaf prediction value."""
        raise NotImplementedError("_leaf_value must be implemented in subclasses.")


class DecisionTreeClassifier(_BaseDecisionTree):
    """
    Decision Tree Classifier.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at a leaf.
    max_features : int or None, default=None
        Number of features per split.
    criterion : {"gini", "entropy"}, default="gini"
        Split quality measure.

    Attributes
    ----------
    root_ : _TreeNode
        Root node of fitted tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    criterion : str
        Criterion used.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Literal["gini", "entropy"] = "gini",
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        if criterion not in ("gini", "entropy"):
            raise ValueError(f"Unknown criterion '{criterion}'. Expected 'gini' or 'entropy'.")

        self.criterion = criterion

    def _impurity(self, y: np.ndarray) -> float:
        """Calculate impurity."""
        values, counts = np.unique(y, return_counts=True)
        p = counts / len(y)

        if self.criterion == "gini":
            return 1 - np.sum(p ** 2)

        elif self.criterion == "entropy":
            return -np.sum(p * np.log2(p + 1e-12))

        else:
            raise ValueError(f"Unknown criterion '{self.criterion}'.")

    def _leaf_value(self, y: np.ndarray) -> float:
        """Determine majority class."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


class DecisionTreeRegressor(_BaseDecisionTree):
    """
    Decision Tree Regressor.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at a leaf.
    max_features : int or None, default=None
        Number of features per split.
    criterion : {"mse"}, default="mse"
        Split quality measure.

    Attributes
    ----------
    root_ : _TreeNode
        Root node of fitted tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    criterion : str
        Criterion used.
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        criterion: Literal["mse"] = "mse",
    ):
        super().__init__(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

        if criterion != "mse":
            raise ValueError(f"Unknown criterion '{criterion}'. Only 'mse' is supported.")

        self.criterion = criterion

    def _impurity(self, y: np.ndarray) -> float:
        """Calculate variance."""
        mu = np.mean(y)
        return np.mean((y - mu) ** 2)

    def _leaf_value(self, y: np.ndarray) -> float:
        """Determine mean value."""
        return float(np.mean(y))