"""
Preprocessing for supervised learning models.

This module implements all functions that prepare the data
for supervised learning. It implements common functions
that are used across the majority of algorithms.

Functions
-------
train_test_split
    Split array-like objects into train and test subsets.
    
normalize
    Normalize the data using your chosen metric.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union
import numbers
from warnings import warn
from ._validation import *


__all__ = [
    'train_test_split',
    'normalize'
]

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]

def train_test_split(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    test_size: float = 0.2,
    shuffle: bool = True,
    stratify: Optional[ArrayLike] = None,
    random_state: Optional[int] = None,
) -> Union[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Split up data into training and testing sets.
    
    Inputs
    -------
    X: ArrayLike
        The features. A mutable array-like type. Can be a list, tuple,
        np.ndarray, or pd.DataFrame.
        Must match length of Y.
    y: ArrayLike or None, optional
        The labels. A mutable array-like type. Can be a list, tuple,
        np.ndarray, or pd.DataFrame.
        Must match length of X.
    test_size: float, optional
        The proportion of the samples that will be testing samples.
        0 < test_size < 1.0
        Default is 0.2
    shuffle: bool, optional
        Whether the testing samples will be selected randomly or from the input ordering.
        Default is True.
    stratify: ArrayLike or None, optional
        If provided, the data is split up within each class.
    random_state: int, optional
        Set a state for consistent results.
        0 < random_state < 2^32
        Optional.

    Returns
    -------
    X_train, X_test : np.ndarray
        Training and testing data.
    y_train, y_test : np.ndarray, optional
        Training and testing labels (only if y is provided).

    Decisions
    -------
    Ceiling the number of test samples as other well-known libraries do for train_test_split functions.
    """
    X = _validate_2d_array(X)
    
    if not (0 < test_size < 1.0):
        raise ValueError("test_size must be a float between 0 and 1 exclusive.")

    if stratify is not None and len(X) != len(stratify):
        raise ValueError("X and stratify must have the same length.")
    
    if not (random_state is None or 0 <= random_state <= 2**32 - 1):
        raise ValueError("random_state must be an integer between 0 and 2^32 -1 inclusive.")

    if y is not None:
        y = np.asarray(y)
        _check_same_length(X, y, "X", "y")
    
    # Split within stratify labels.
    if stratify is not None:
        stratify = np.asarray(stratify)

        # Get unique labels and their corresponding indices.
        unique = np.unique(stratify)
        label_to_idx = {label: np.where(stratify == label)[0] for label in unique}

        # Split up within each label to build the train and test indices.
        train_idxs = []
        test_idxs = []
        rng = np.random.default_rng(random_state)
        for _, idxs in label_to_idx.items():
            n = len(idxs)

            # Number of testing samples.
            test_quant = int(np.ceil(n * test_size))
            if shuffle:
                rng.shuffle(idxs)
            train_idxs.append(idxs[:-test_quant])
            test_idxs.append(idxs[-test_quant:])

        # Flatten idx arrays.
        train_idxs = np.concatenate(train_idxs)
        test_idxs = np.concatenate(test_idxs)
        X_train = X[train_idxs]
        X_test = X[test_idxs]
        if y is not None:
            y_train = y[train_idxs]
            y_test = y[test_idxs]
            return (X_train, X_test, y_train, y_test)
        return (X_train, X_test)
    else: # Split normally (no stratify).
        idxs = np.arange(len(X))

        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(idxs)
        test_quant = int(np.ceil(len(X) * test_size))
        train_idxs = idxs[:-test_quant]
        test_idxs = idxs[-test_quant:]

        if y is not None:
            X_train = X[train_idxs]
            X_test = X[test_idxs]
            y_train = y[train_idxs]
            y_test = y[test_idxs]
            return (X_train, X_test, y_train, y_test)
        X_train = X[train_idxs]
        X_test = X[test_idxs]
        return (X_train, X_test)

def normalize(
    x: ArrayLike,
    method: str = "zscore",
    axis: Optional[int] = None,
    feature_range: Optional[tuple[Union[float, int], Union[float, int]]] = None,
    stats: Optional[dict] = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """
    Normalizes an array using one of several normalization methods.

    Stateful methods (zscore, minmax, robust) may return statistics
    computed from the data and reuse them to avoid data leakage.

    Stateless methods (l1, l2) do not support statistics.
    """

    x = _validate_1d_or_2d(x)
    axis = _validate_axis(axis, x.ndim)

    if feature_range is not None and method != "minmax":
        warn(
            f"'feature_range' was provided but will be ignored since method is not 'minmax'.",
            UserWarning,
        )

    match method:
        case "zscore":
            if stats is None:
                mean = x.mean(axis=axis, keepdims=True)
                std = x.std(axis=axis, keepdims=True)
                std = np.where(std == 0, 1.0, std)
                stats = {"mean": mean, "std": std}
            else:
                mean = stats["mean"]
                std = stats["std"]

            x_norm = (x - mean) / std
            return (x_norm, stats) if return_stats else x_norm

        case "minmax":
            if feature_range is not None:
                if not isinstance(feature_range, tuple) or len(feature_range) != 2:
                    raise TypeError("'feature_range' must be a tuple (min, max)")
                if feature_range[0] >= feature_range[1]:
                    raise ValueError("feature_range min must be < max")
                if not all(isinstance(v, numbers.Real) for v in feature_range):
                    raise TypeError("'feature_range' must contain only numeric values")
                a, b = map(float, feature_range)
            else:
                a, b = 0.0, 1.0

            if stats is None:
                x_min = x.min(axis=axis, keepdims=True)
                x_max = x.max(axis=axis, keepdims=True)
                denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min)
                stats = {"min": x_min, "denom": denom, "range": (a, b)}
            else:
                x_min = stats["min"]
                denom = stats["denom"]
                a, b = stats["range"]

            x_norm = a + ((x - x_min) / denom) * (b - a)
            return (x_norm, stats) if return_stats else x_norm

        case "robust":
            if stats is None:
                med = np.median(x, axis=axis, keepdims=True)
                q75 = np.percentile(x, 75, axis=axis, keepdims=True)
                q25 = np.percentile(x, 25, axis=axis, keepdims=True)
                denom = np.where(q75 - q25 == 0, 1.0, q75 - q25)
                stats = {"median": med, "denom": denom}
            else:
                med = stats["median"]
                denom = stats["denom"]

            x_norm = (x - med) / denom
            return (x_norm, stats) if return_stats else x_norm

        case "l1":
            denom = np.sum(np.abs(x), axis=axis, keepdims=True)
            denom = np.where(denom == 0, 1.0, denom)
            return x / denom

        case "l2":
            denom = np.sqrt(np.sum(x ** 2, axis=axis, keepdims=True))
            denom = np.where(denom == 0, 1.0, denom)
            return x / denom

        case _:
            raise ValueError(
                "method must be one of: zscore, minmax, robust, l1, l2"
            )