import numpy as np
from typing import Optional, Literal

from rice_ml.utilities import *
from rice_ml.utilities._validation import *


class _TreeNode:
    """
    Internal node structure for decision tree.

    Parameters
    ----------
    feature : int or None, default=None
        Index of the feature to split on.
    threshold : float or None, default=None
        Threshold value for the split.
    left : _TreeNode or None, default=None
        Left child node.
    right : _TreeNode or None, default=None
        Right child node.
    value : float or None, default=None
        Prediction value if this is a leaf node.
    """

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
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int or None, default=None
        Number of features to consider when looking for the best split.

    Attributes
    ----------
    root_ : _TreeNode
        The root node of the fitted tree.
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

    def fit(self, X, y):
        """
        Build a decision tree from the training set.

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
        y = y.astype(int)

        _check_same_length(X, y, "X", "y")
        _check_finite_if_numeric(X)
        _check_finite_if_numeric(y)

        self.n_samples_, self.n_features_ = X.shape

        root = self._build_tree(X, y, depth=0)

        if root is None:
            raise RuntimeError("Tree failed to build: _build_tree returned None.")

        self.root_ = root

        return self


    def predict(self, X):
        """
        Predict class labels or regression values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (for classifier) or values (for regressor).
        """
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("DecisionTree model has not been fit yet.")

        X = _validate_2d_array(X)
        _check_finite_if_numeric(X)

        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}."
            )

        preds = np.array([self._traverse_tree(x, self.root_) for x in X])

        return preds
    

    def score(self, X, y):
        """
        Return the score of the predictions on the given test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels or values for X.

        Returns
        -------
        float
            Mean accuracy (for classifier) or R^2 score (for regressor).
        """
        y = _validate_1d_array(y)
        _check_finite_if_numeric(y)

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

    def _build_tree(self, X, y, depth: int):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data for this node.
        y : ndarray of shape (n_samples,)
            Target values for this node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        _TreeNode
            Root node of the subtree.
        """
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

    def _best_split(self, X, y):
        """
        Find the best split for the given data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        tuple of (int, float) or None
            Best feature index and threshold value, or None if no valid split found.
        """
        n_samples, n_features = X.shape
        best_gain = -np.inf
        best_split = None

        parent_impurity = self._impurity(y)

        if self.max_features is None:
            features = range(n_features)
        else:
            rng = np.random.default_rng(42)
            features = rng.choice(
                n_features,
                size=min(self.max_features, n_features),
                replace=False,
            )

        for feature in features:
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

    def _split(self, X, y, feature: int, threshold: float):
        """
        Split data based on a feature and threshold.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        feature : int
            Feature index to split on.
        threshold : float
            Threshold value for the split.

        Returns
        -------
        tuple of (ndarray, ndarray, ndarray, ndarray)
            X_left, y_left, X_right, y_right after splitting.
        """
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        return (
            X[left_mask], y[left_mask],
            X[right_mask], y[right_mask],
        )

    def _traverse_tree(self, x, node: _TreeNode):
        """
        Traverse the tree to make a prediction for a single sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Single sample to predict.
        node : _TreeNode
            Current node in the tree.

        Returns
        -------
        float
            Predicted value.
        """
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


    def get_depth(self):
        """
        Get the depth of the fitted decision tree.

        Returns
        -------
        int
            Maximum depth of the tree.
        """
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("DecisionTree model has not been fit yet.")
        return self._get_depth(self.root_)

    def _get_depth(self, node: _TreeNode):
        """
        Recursively compute the depth of a subtree.

        Parameters
        ----------
        node : _TreeNode
            Root of the subtree.

        Returns
        -------
        int
            Depth of the subtree.
        """
        if node is None:
            return 0

        if node.value is not None:
            return 1

        left_depth = self._get_depth(node.left) if node.left is not None else 0
        right_depth = self._get_depth(node.right) if node.right is not None else 0

        return 1 + max(left_depth, right_depth)


    def get_n_leaves(self):
        """
        Get the number of leaves in the fitted decision tree.

        Returns
        -------
        int
            Number of leaf nodes.
        """
        if not hasattr(self, "root_") or self.root_ is None:
            raise RuntimeError("DecisionTree model has not been fit yet.")
        return self._get_n_leaves(self.root_)

    def _get_n_leaves(self, node: _TreeNode):
        """
        Recursively count the number of leaves in a subtree.

        Parameters
        ----------
        node : _TreeNode
            Root of the subtree.

        Returns
        -------
        int
            Number of leaf nodes in the subtree.
        """
        if node is None:
            return 0

        if node.value is not None:
            return 1
        
        left_leaves = self._get_n_leaves(node.left) if node.left is not None else 0
        right_leaves = self._get_n_leaves(node.right) if node.right is not None else 0

        return left_leaves + right_leaves

    def _impurity(self, y):
        """
        Calculate impurity of target values.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        float
            Impurity measure.
        """
        raise NotImplementedError("_impurity must be implemented in subclasses.")

    def _leaf_value(self, y):
        """
        Determine the prediction value for a leaf node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values at the leaf.

        Returns
        -------
        float
            Prediction value for the leaf.
        """
        raise NotImplementedError("_leaf_value must be implemented in subclasses.")


class DecisionTreeClassifier(_BaseDecisionTree):
    """
    Decision Tree Classifier using CART algorithm.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure
        or contain fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int or None, default=None
        Number of features to consider when looking for the best split.
    criterion : {'gini', 'entropy'}, default='gini'
        Function to measure the quality of a split.

    Attributes
    ----------
    root_ : _TreeNode
        The root node of the fitted tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    criterion : str
        The criterion used to measure split quality.
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

    def _impurity(self, y):
        """
        Calculate impurity using Gini index or entropy.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        float
            Impurity value (Gini index or entropy).
        """
        values, counts = np.unique(y, return_counts=True)
        p = counts / len(y)

        if self.criterion == "gini":
            return 1 - np.sum(p ** 2)

        elif self.criterion == "entropy":
            return -np.sum(p * np.log2(p + 1e-12))

        else:
            raise ValueError(f"Unknown criterion '{self.criterion}'.")

    def _leaf_value(self, y):
        """
        Determine the majority class for a leaf node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target class labels at the leaf.

        Returns
        -------
        float
            Majority class label.
        """
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


class DecisionTreeRegressor(_BaseDecisionTree):
    """
    Decision Tree Regressor using CART algorithm.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until all leaves contain
        fewer than min_samples_split samples.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int or None, default=None
        Number of features to consider when looking for the best split.
    criterion : {'mse'}, default='mse'
        Function to measure the quality of a split. Only 'mse' (mean squared error) is supported.

    Attributes
    ----------
    root_ : _TreeNode
        The root node of the fitted tree.
    n_samples_ : int
        Number of samples seen during fit.
    n_features_ : int
        Number of features seen during fit.
    criterion : str
        The criterion used to measure split quality.

    Note:
    This is also implemented in regression_trees.py. They are
    implemented slightly differently, because in this class it uses
    a base class as well to reduce repetitive code between this class
    and the classifier version of it.

    In the regression_trees class, it is built contained completely in
    that file/class. While they do the same thing, they are both implemented
    for ease of access for users.
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


    def _impurity(self, y):
        """
        Calculate variance (mean squared error from mean) for regression.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        float
            Variance of the target values.
        """
        mu = np.mean(y)
        return np.mean((y - mu) ** 2)

    def _leaf_value(self, y):
        """
        Determine the mean value for a leaf node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values at the leaf.

        Returns
        -------
        float
            Mean of the target values.
        """
        return float(np.mean(y))