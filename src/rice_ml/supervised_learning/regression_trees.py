import numpy as np
from typing import Optional, Literal

from src.rice_ml.utilities import *
from src.rice_ml.utilities._validation import *

class _TreeNode:
    """
    Internal node structure for decision tree.

    Parameters
    ----------
    feature_idx : int or None, optional
        Index of the feature used for splitting at this node.
    threshold : float or None, optional
        Threshold value for the split.
    left : _TreeNode or None, optional
        Left child node (samples <= threshold).
    right : _TreeNode or None, optional
        Right child node (samples > threshold).
    value : float or None, optional
        Predicted value for leaf nodes (mean of target values).
    """
    def __init__(
        self,
        *,
        feature_idx=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # float output (mean of y)


class DecisionTreeRegressor:
    """
    CART-style decision tree regressor using variance reduction.

    A decision tree regressor that recursively partitions the feature space
    to minimize the variance of target values within each partition. Uses
    the mean squared error (MSE) reduction criterion for splitting decisions.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum depth of the tree. If None, nodes are expanded until all
        leaves contain fewer than min_samples_split samples or are pure.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    min_samples_leaf : int, default=1
        Minimum number of samples required to be at a leaf node.
    max_features : int, float, or None, default=None
        Number of features to consider when looking for the best split:
        - If int, consider `max_features` features at each split.
        - If float, consider `int(max_features * n_features)` features.
        - If None, consider all features.
    random_state : int or None, default=None
        Controls the randomness of the feature sampling when max_features
        is not None. Pass an int for reproducible output across function calls.

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

    Notes
    -----
    Decision trees can overfit easily on training data, especially when
    max_depth is not limited. Consider using ensemble methods like Random
    Forest or Gradient Boosting for better generalization.
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

    def fit(self, X, y):
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values (real numbers).

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If X and y have different lengths.
        """
        X = _validate_2d_array(X)
        y = _validate_1d_array(y)
        _check_same_length(X, y, "X", "y")
        _check_finite_if_numeric(X)
        _check_finite_if_numeric(y)

        self.n_features_in = X.shape[1]
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data subset for this node.
        y : ndarray of shape (n_samples,)
            Target values for this node.
        depth : int
            Current depth in the tree.

        Returns
        -------
        _TreeNode
            The constructed tree node (internal or leaf).
        """
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
    
    def _make_leaf(self, y):
        """
        Create a leaf node with the mean of target values.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values for this leaf.

        Returns
        -------
        _TreeNode
            Leaf node with predicted value set to mean(y).
        """
        return _TreeNode(value=float(np.mean(y)))
    
    def _variance(self, y):
        """
        Compute total variance (sum of squared deviations).

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        float
            Total variance (variance * n_samples).
        """
        return np.var(y) * len(y)
    
    def _best_split(self, X, y, features):
        """
        Find the best split for the given data and feature subset.

        Searches over all candidate thresholds for each feature to find
        the split that minimizes the total variance of the resulting
        child nodes.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        features : array-like of int
            Indices of features to consider for splitting.

        Returns
        -------
        dict or None
            Dictionary containing split information:
            - 'loss': Total variance after split
            - 'feature': Feature index for split
            - 'threshold': Threshold value for split
            - 'left': Tuple of (X_left, y_left)
            - 'right': Tuple of (X_right, y_right)
            Returns None if no valid split is found.
        """
        best = {
            "loss": np.inf,
            "feature": None,
            "threshold": None,
            "left": None,     # (X_left, y_left)
            "right": None,    # (X_right, y_right)
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

        # no valid split  
        if best["feature"] is None:
            return None

        return best

    
    def _get_feature_subset(self, d):
        """
        Sample a subset of features for splitting.

        Parameters
        ----------
        d : int
            Total number of features.

        Returns
        -------
        range or ndarray
            Indices of features to consider for splitting.

        Raises
        ------
        ValueError
            If max_features has an invalid type.
        """
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
    
    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If X has a different number of features than seen during fit.
        """
        if not hasattr(self, "root_"):
            raise RuntimeError("Model not fit yet.")

        X = _validate_2d_array(X)
        _check_finite_if_numeric(X)

        if X.shape[1] != self.n_features_in:
            raise ValueError("Feature mismatch.")

        return np.array([self._predict_row(x, self.root_) for x in X])
    
    def _predict_row(self, x, node):
        """
        Predict a single sample by traversing the tree.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Single sample to predict.
        node : _TreeNode
            Current node in the tree.

        Returns
        -------
        float
            Predicted value for the sample.
        """
        if node.value is not None:
            return node.value

        if x[node.feature_idx] <= node.threshold:
            return self._predict_row(x, node.left)
        else:
            return self._predict_row(x, node.right)

    def score(self, X, y):
        """
        Return the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the residual
        sum of squares ((y_true - y_pred)^2).sum() and v is the total sum of
        squares ((y_true - y_true.mean())^2).sum(). The best possible score
        is 1.0 and it can be negative (because the model can be arbitrarily
        worse).

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
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot