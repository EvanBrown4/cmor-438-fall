"""
Preprocessing for supervised learning models.

This module implements all functions that prepare the data
for supervised learning. It implements common functions
that are used across the majority of algorithms.

Functions
-------
train_test_split
    Split array-like objects into train and test subsets.
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, Literal
from warnings import warn

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
        Must match length/size of Y.
    y: ArrayLike or None, optional
        The labels. A mutable array-like type. Can be a list, tuple,
        np.ndarray, or pd.DataFrame.
        Must match length/size of X.
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
    X = np.asarray(X)
    x_len = len(X) # Stored here to avoid continuous querying of length.

    ## Type/Value checking.
    if (x_len == 0):
        raise ValueError("With n_samples=%d and test_size=%d, the resulting train set will be empty.", 0, test_size)
    
    if not (0 < test_size < 1.0):
        raise ValueError("test_size must be a float between 0 and 1 exclusive.")

    if stratify is not None and x_len != len(stratify):
        raise ValueError("X and stratify must have the same length.")
    
    if not (random_state is None or 0 <= random_state <= 2**32 - 1):
        raise ValueError("random_state must be an integer between 0 and 2^32 -1 inclusive.")

    if y is not None:
        y = np.asarray(y)
        if x_len != len(y):
            raise ValueError("X and Y must have the same length.")
    
    # Split within stratify labels.
    if stratify is not None:
        stratify = np.asarray(stratify)

        # Get unique labels and their corresponding indices.
        unique = np.unique(stratify)
        label_to_idx = {label: np.where(stratify == label)[0] for label in unique}

        # Split up within each label to build the train and test indices.
        train_idxs = []
        test_idxs = []
        for _, idxs in label_to_idx.items():
            n = len(idxs)

            # Number of testing samples.
            test_quant = int(np.ceil(n * test_size))
            if shuffle:
                rng = np.random.default_rng(random_state)
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
        idxs = np.arange(x_len)

        if shuffle:
            rng = np.random.default_rng(random_state)
            rng.shuffle(idxs)
        test_quant = int(np.ceil(x_len * test_size))
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
        axis: Optional[Literal[0,1]] = None,
        feature_range: Optional[tuple[Union[float, int], Union[float, int]]] = None
) -> np.ndarray:
    """
    Normalizes an array using one of several normalization methods.
    Supports z-score, min-max, robust, L1, and L2 normalization.

    Inputs
    -------
    x: ArrayLike - element type must be a number (floats or ints).
        The array to be normalized. Can be 1-dimensional or 2-dimensional.

    method: "zscore" | "minmax" | "robust" | "l1" | "l2"
        The method x will be normalized with.
        Default: zscore.

    axis: 0 | 1 | None
        The axis to normalize on. Cannot be 1 for 1d arrays.
        Default: None. Automatically set to 0 for 2D arrays (normalize per feature).

    feature_range: tuple(float | int, float | int), optional
        Used only when method="minmax"
        The desired output range (min, max) for the scaled data.
        Default: None. Automatically set to (0.0, 1.0) if method="minmax"

    Output
    -------
    The normalized version of x.

    Examples
    -------
    >>> import numpy as np
    >>> x = np.array([[1,2], [3,4]])
    >>> normalize(x, method="zscore")
    array([[-1.0, -1.0],
           [ 1.0,  1.0]])
    """

    # Ensure valid axis.
    if axis not in (0, 1, None):
        raise ValueError("'axis' must be 0, 1, or None")
    
    x = np.asarray(x)

    # Ensure valid size/length of x and valid element typing.
    if len(x) == 0 or x.size == 0:
        raise ValueError("Cannot normalize a 0-length array or 0-size matrix.")

    if not(np.issubdtype(x.dtype, np.floating) or np.issubdtype(x.dtype, np.integer)):
        raise ValueError("Cannot normalize a non-number array. At least one element is not a number.")

    # Convert x elements to floats.
    x = x.astype(float)

    ## Dimensionalit and axis checks.
    if x.ndim > 2 or x.ndim < 1:
        raise ValueError("normalize currently supports only 1D or 2D arrays.")

    if x.ndim == 1 and axis == 1:
        raise ValueError("Axis cannot be 1 for a 1-dimensional array.")
    
    if axis is None:
        axis = 0 if x.ndim == 2 else None
    
    match method:
        case "zscore":
            return (x - x.mean(axis=axis)) / x.std(axis=axis)
        case "minmax":
            if feature_range is not None:
                if method != "minmax":
                    warn(
                        f"'feature_range' was provided but will be ignored since method='{method}'. It is only used for 'minmax' normalization.",
                        UserWarning
                    )
                if not isinstance(feature_range, tuple) or len(feature_range) != 2:
                    raise TypeError("'feature_range' must be a tuple of two numbers (min, max)")
                if not all(isinstance(v, (int, float)) for v in feature_range):
                    raise TypeError("'feature_range' values must be numeric (int or float)")
                
                
                if feature_range[0] >= feature_range[1]:
                    raise ValueError(
                        f"Min value for feature_range must be less than max value. Input: {feature_range}"
                    )
            else:
                feature_range = (0.0, 1.0)
            
            # Convert range boundaries to floats.
            a, b = map(float, feature_range)

            x_min = x.min(axis=axis, keepdims=True)
            x_max = x.max(axis=axis, keepdims=True)
            denom = np.where(x_max - x_min == 0, 1.0, x_max - x_min) # Divide by zero prevention.
            return a + ((x - x_min) / denom) * (b-a)
        case "robust":
            x_med = np.median(x, axis=axis, keepdims=True)
            x_75 = np.percentile(x, q=75, axis=axis, keepdims=True)
            x_25 = np.percentile(x, q=25, axis=axis, keepdims=True)
            denom = np.where(x_75 - x_25 == 0, 1.0, x_75 - x_25) # Divide by zero prevention.
            return (x - x_med) / (denom)
        case "l1":
            denom = np.sum(np.abs(x), axis=axis, keepdims=True)
            denom = np.where(denom == 0, 1.0, denom) # Divide by zero prevention.
            return x / denom
        case "l2":
            sum_sq = np.sum(x**2, axis=axis, keepdims=True)
            sum_sq = np.where(sum_sq == 0, 1.0, sum_sq) # Divide by zero prevention.
            return x / np.sqrt(sum_sq)
        case _:
            raise ValueError(f"'method' must be one of: zscore, minmax, robust, l1, l2. Input was {method}")


# def scale(x: np.array):
#     pass