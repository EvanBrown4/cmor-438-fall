import numpy as np
import pandas as pd
from typing import Optional, Union, Literal

__all__ = [
    "_validate_1d_array",
    "_validate_2d_array",
    "_check_same_length",
    "_check_same_shape",
    "_validate_axis",
    "_validate_1d_or_2d",
]

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]

def _check_same_length(a: ArrayLike, b: ArrayLike, a_name: str, b_name: str) -> None:
    a_len = len(a)
    b_len = len(b)
    if a_len != b_len:
        raise ValueError(f"{a_name} and {b_name} must have the same length, "
                         f"got {a_len} and {b_len}.")
    

def _check_same_shape(a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{a_name} and {b_name} must have the same shape, "
                         f"got {a.shape} and {b.shape}.")

def _validate_1d_or_2d(arr: ArrayLike) -> np.ndarray:
    """
    Attempt to validate arr as 1D OR 2D.
    Returns the cleaned numpy array.
    Only raises an error if BOTH 1D and 2D validation fail.
    """

    # Try 1D
    try:
        return _validate_1d_array(arr)
    except Exception as e1:
        err_1d = e1

    # Try 2D
    try:
        return _validate_2d_array(arr)
    except Exception as e2:
        err_2d = e2
    
    # If both fail, raise a combined error
    raise ValueError(
        "Input must be a 1D or 2D numeric array.\n\n"
        f"1D validation error: {err_1d}\n"
        f"2D validation error: {err_2d}"
    )

def _validate_1d_array(y: ArrayLike) -> np.ndarray:
    """
    Convert input into a clean, 1D, numeric numpy array of floats.

    Input:
    An array like object.
    
    Ensures:
    - Exactly 1 dimension
    - Not empty
    - Numeric dtype
    - No infinite or NaNs

    Returns
    The input converted to a clean numpy array of floats.
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError(f"Expected 1D. Got {y.ndim} dimensions.")

    ## Type/Value checking.
    if (y.size == 0):
        raise ValueError("Array cannot be empty.")

    if y.dtype == object:
        if not all(isinstance(v, (int, float, np.number)) for v in y.flat):
            raise TypeError("Array contains non-numeric values.")
    elif not np.issubdtype(y.dtype, np.number):
        raise TypeError("Array must contain numeric values.")
    y = y.astype(float)
    
    if not np.all(np.isfinite(y)):
        raise ValueError("Array must not contain NaN or infinite values.")
    
    return y

def _validate_2d_array(x: ArrayLike) -> np.ndarray:
    """
    Convert input into a clean, 2D, numeric numpy array of floats.

    Input:
    An array like object.
    
    Ensures:
    - Exactly 2 dimensions
    - Not empty
    - Numeric dtype
    - No infinite or NaNs

    Returns
    The input converted to a clean numpy array of floats.
    """
    x = np.asarray(x)

    if x.ndim != 2:
        raise ValueError(f"Expected 2D. Got {x.ndim} dimensions.")

    ## Type/Value checking.
    if (x.size == 0):
        raise ValueError("Array cannot be empty.")

    if x.dtype == object:
        if not all(isinstance(v, (int, float, np.number)) for v in x.flat):
            raise TypeError("Array contains non-numeric values.")
    elif not np.issubdtype(x.dtype, np.number):
        raise TypeError("Array must contain numeric values.")
    x = x.astype(float)
    
    if not np.all(np.isfinite(x)):
        raise ValueError("Array must not contain NaN or infinite values.")
    
    return x

def _validate_axis(axis: Optional[int], ndim: int) -> Optional[int]:
    if axis is None:
        if ndim > 1:
            return 0
        return None
    if not isinstance(axis, int):
        raise TypeError("'axis' must be an integer or None.")
    if axis >= ndim or axis < -ndim:
        raise ValueError(f"'axis'={axis} out of bounds for array with {ndim} dimensions.")
    return axis