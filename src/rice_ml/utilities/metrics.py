"""
metrics.py
Contains any necessary metric computations.

euclidean_dist: Computes the euclidean distance (l2) of two numpy arrays.
manhattan_dist: Computes the manahattan distance (l1) of two numpy arrays.
"""

import numpy as np
from typing import Union, Optional
import pandas as pd

from ._validation import *

__all__ = [
    'euclidean_dist',
    'manhattan_dist',
    'r2_score',
]

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]

def euclidean_dist(x: ArrayLike,
                   y: ArrayLike,
                   axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Computes the euclidean (l2) distance of two arrays.

    Inputs
    -------
    x: ArrayLike - element type must be a number (floats or ints).
        The array to be normalized. Can be 1-dimensional or 2-dimensional.
        
    y: ArrayLike - element type must be a number (floats or ints).
        The array to be normalized. Can be 1-dimensional or 2-dimensional.

    axis: int, optional
        Axis along which to compute distances.
        - For 1D arrays → return a scalar distance.
        - For 2D arrays → default to axis=0 (compute by columns).
        Axis must be within [-ndim, ndim-1].

    Output
    -------
    The euclidean distance between x and y.

    Constraints
    -------
    x and y must have the same shape.

    Examples
    -------
    >>> import numpy as np
    >>> x = np.array([[0,0], [0,1]])
    >>> y = np.array([[1,1], [1,2]])
    >>> euclidean_dist(x, y)

    """
    x = _validate_1d_or_2d(x)
    y = _validate_1d_or_2d(y)

    _check_same_shape(x, y, "x", "y")
    # _check_same_length(x, y, "x", "y")
    
    if axis is None:
        if x.ndim == 1:
            return float(np.sqrt(np.sum((x - y) ** 2)))
        else:
            axis = 0
    else:
        if not isinstance(axis, int):
            raise TypeError("'axis' must be an integer or None.")
        if axis >= x.ndim or axis < -x.ndim:
            raise ValueError(f"'axis'={axis} out of bounds for array with {x.ndim} dims.")

    return np.sqrt(np.sum((x-y)**2, axis=axis))

def manhattan_dist(x: ArrayLike,
                   y: ArrayLike,
                   axis: Optional[int] = None
) -> Union[float, np.ndarray]:
    """
    Computes the manhattan (l1) distance of two arrays.

    Inputs
    -------
    x: ArrayLike - element type must be a number (floats or ints).
        The array to be normalized. Can be 1-dimensional or 2-dimensional.
        
    y: ArrayLike - element type must be a number (floats or ints).
        The array to be normalized. Can be 1-dimensional or 2-dimensional.

    axis: int, optional
        Axis along which to compute distances.
        - For 1D arrays → return a scalar distance.
        - For 2D arrays → default to axis=0 (compute by columns).
        Axis must be within [-ndim, ndim-1].

    Output
    -------
    The manhattan distance between x and y.

    Constraints
    -------
    x and y must have the same shape.

    Examples
    -------
    >>> import numpy as np
    >>> x = np.array([[0,0], [0,1]])
    >>> y = np.array([[1,1], [1,2]])
    >>> manhattan_dist(x, y)

    """
    x = _validate_1d_or_2d(x)
    y = _validate_1d_or_2d(y)

    _check_same_shape(x, y, "x", "y")
    # _check_same_length(x, y, "x", "y")
    
    if axis is None:
        if x.ndim == 1:
            return float(np.sum(np.abs(x - y)))
        else:
            axis = 0
    else:
        if not isinstance(axis, int):
            raise TypeError("'axis' must be an integer or None.")
        if axis >= x.ndim or axis < -x.ndim:
            raise ValueError(f"'axis'={axis} out of bounds for array with {x.ndim} dims.")
    return np.sum(np.abs(x - y), axis=axis)

def r2_score(y_true, y_pred):
    """
    Compute the coefficient of determination (R^2).

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    float
        R^2 score. Best possible score is 1.0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if np.isclose(ss_res, 0.0) else 0.0

    return 1 - ss_res / ss_tot