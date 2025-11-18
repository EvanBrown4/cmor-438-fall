"""
metrics.py
Contains any necessary metric computations.

euclidean_dist: Computes the euclidean distance (l2) of two numpy arrays.
manhattan_dist: Computes the manahattan distance (l1) of two numpy arrays.
"""

import numpy as np
from typing import Union, Optional
import pandas as pd

__all__ = [
    'euclidean_dist',
    'manhattan_dist'
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
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 0 or y.ndim == 0:
        raise ValueError("Cannot pass scalar inputs; expected 1D or 2D arrays.")
    
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("Must pass arrays of 1-dimension or 2-dimensions.")

    ## Type/Value checking.
    if (x.size == 0):
        raise ValueError("Input x cannot be empty.")
    
    if (y.size == 0):
        raise ValueError("Input y cannot be empty.")
    
    if len(x) != len(y):
        raise ValueError(f"Arrays must have the same shape, got {x.shape} and {y.shape}")
    
    if x.shape != y.shape:
        raise ValueError(f"Arrays must have the same shape, got {x.shape} and {y.shape}")
    
    if x.dtype == object:
        if not all(isinstance(v, (int, float, np.number)) for v in x.flat):
            raise TypeError("x contains non-numeric values.")
    elif not np.issubdtype(x.dtype, np.number):
        raise TypeError("x must contain numeric values.")

    if y.dtype == object:
        if not all(isinstance(v, (int, float, np.number)) for v in y.flat):
            raise TypeError("y contains non-numeric values.")
    elif not np.issubdtype(y.dtype, np.number):
        raise TypeError("y must contain numeric values.")
    
    x = x.astype(float)
    y = y.astype(float)
    
    if not np.all(np.isfinite(x)):
        raise ValueError("Inputs must not contain NaN or infinite values. Failing array: x")
    
    if not np.all(np.isfinite(y)):
        raise ValueError("Inputs must not contain NaN or infinite values. Failing array: y")
    
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
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim == 0 or y.ndim == 0:
        raise ValueError("Cannot pass scalar inputs; expected 1D or 2D arrays.")
    
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("Must pass arrays of 1-dimension or 2-dimensions.")

    ## Type/Value checking.
    if (x.size == 0):
        raise ValueError("Input x cannot be empty.")
    
    if (y.size == 0):
        raise ValueError("Input y cannot be empty.")
    
    if len(x) != len(y):
        raise ValueError(f"Arrays must have the same shape, got {x.shape} and {y.shape}")
    
    if x.shape != y.shape:
        raise ValueError(f"Arrays must have the same shape, got {x.shape} and {y.shape}")
    
    if x.dtype == object:
        if not all(isinstance(v, (int, float, np.number)) for v in x.flat):
            raise TypeError("x contains non-numeric values.")
    elif not np.issubdtype(x.dtype, np.number):
        raise TypeError("x must contain numeric values.")

    if y.dtype == object:
        if not all(isinstance(v, (int, float, np.number)) for v in y.flat):
            raise TypeError("y contains non-numeric values.")
    elif not np.issubdtype(y.dtype, np.number):
        raise TypeError("y must contain numeric values.")
    
    x = x.astype(float)
    y = y.astype(float)
    
    if not np.all(np.isfinite(x)):
        raise ValueError("Inputs must not contain NaN or infinite values. Failing array: x")
    
    if not np.all(np.isfinite(y)):
        raise ValueError("Inputs must not contain NaN or infinite values. Failing array: y")
    
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