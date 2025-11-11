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
from typing import Optional, Union

__all__ = [
    'train_test_split'
]

ArrayLike = Union[list, tuple, np.ndarray, pd.DataFrame]

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

def normalize(x: np.array):
    """
    Normalizes the given array using its elements' z-scores.

    Inputs:
    x: A numpy array. Type must be a Number (floats or ints).

    Output:
    The z-score normalized version of x.
    """
    mean = np.mean(x)
    stddev = np.std(x)
    return (x - mean) / stddev


def scale(x: np.array):
    pass