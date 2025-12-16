import numpy as np
import pandas as pd
from typing import Optional, Literal, Union

from rice_ml.utilities._validation import (
    _validate_2d_array,
    _validate_1d_array,
    _check_same_length,
)
from rice_ml.utilities import r2_score

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class _TreeNode:
    """Internal node structure for decision tree."""
    def __init__(
        self,
        *,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional["_TreeNode"] = None,
        right: Optional["_TreeNode"] = None,
        value: Optional[float] = None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value



class DecisionTreeRegressor:
    """
    CART-style decision tree regressor using variance reduction.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum tree depth.
    min_samples_split : int, default=2
        Minimum samples to split a node.
    min_samples_leaf : int, default=1
        Minimum samples at a leaf.
    max_features : int, float, or None, default=None
        Number of features per split.
    random_state : int or None, default=None
        Random seed for feature sampling.

    Attributes
    ----------
    n_features_in : int
        Number of features seen during fit.
    root_ : _TreeNode
        The root node of the fitted tree.

    Examples
    --------
    >>> from src.rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor
    >>> import numpy as np
    >>> X = np.array([[0], [1], [2], [3]])
    >>> y = np.array([0.0, 1.0, 2.0, 3.0])
    >>> reg = DecisionTreeRegressor(max_depth=2)
    >>> reg.fit(X, y)
    >>> reg.predict([[1.5]])
    array([1.5])

    Note:
    This is also implemented in ensemble_methods.py. They are
    implemented slightly differently, because in that class it uses
    a base class as well to reduce repetitive code between it and the classifier version of it.

    In this class on the other hand, it is built contained completely in
    that file/class. While they do the same thing, they are both implemented
    for ease of access for users.
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTreeRegressor":
        """
        Build decision tree regressor.

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

        self.n_features_in = X.shape[1]
        self.root_: _TreeNode = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _TreeNode:
        """Recursively build tree."""
        n, d = X.shape

        if (
            n < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return self._make_leaf(y)

        feat_idxs = self._get_feature_subset(d)

        best = self._best_split(X, y, feat_idxs)
        if best is None:
            return self._make_leaf(y)

        Xl, yl = best["left"]
        Xr, yr = best["right"]

        left = self._build_tree(Xl, yl, depth + 1)
        right = self._build_tree(Xr, yr, depth + 1)

        return _TreeNode(
            feature_idx=best["feature"],
            threshold=best["threshold"],
            left=left,
            right=right,
        )
    
    def _make_leaf(self, y: np.ndarray) -> _TreeNode:
        """Create leaf node."""
        return _TreeNode(value=float(np.mean(y)))
    
    def _variance(self, y: np.ndarray) -> float:
        """Compute total variance."""
        return np.var(y) * len(y)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray, features):
        """Find best split."""
        best = {
            "loss": np.inf,
            "feature": None,
            "threshold": None,
            "left": None,
            "right": None,
        }

        for f in features:
            thresholds = np.unique(X[:, f])

            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = ~left_mask

                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue

                yL = y[left_mask]
                yR = y[right_mask]

                loss = self._variance(yL) + self._variance(yR)

                if loss < best["loss"]:
                    best["loss"] = loss
                    best["feature"] = f
                    best["threshold"] = t
                    best["left"] = (X[left_mask], yL)
                    best["right"] = (X[right_mask], yR)

        if best["feature"] is None:
            return None

        return best
    
    def _get_feature_subset(self, d: int):
        """Sample feature subset."""
        rng = np.random.default_rng(self.random_state)

        if self.max_features is None:
            return range(d)

        if isinstance(self.max_features, int):
            k = self.max_features
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * d))
        else:
            raise ValueError("Invalid max_features type.")

        return rng.choice(d, size=k, replace=False)
    
    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict regression values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, "root_"):
            raise RuntimeError("Model not fit yet.")

        X = _validate_2d_array(X)

        if X.shape[1] != self.n_features_in:
            raise ValueError("Feature mismatch.")

        return np.array([self._predict_row(x, self.root_) for x in X])
    
    def _predict_row(self, x: np.ndarray, node: _TreeNode) -> float:
        if node.value is not None:
            return node.value

        assert node.left is not None
        assert node.right is not None
        assert node.feature_idx is not None
        assert node.threshold is not None

        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)


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