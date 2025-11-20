"""
postprocess.py

Implements decision making for an ML model.
majority_label() is implemented for classified models.
average_label() is implemented for non-classified models.
"""
import numpy as np
import numpy as np
import pandas as pd
from typing import Literal, Optional, Union

from ._validation import *

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]

def majority_label(labels: ArrayLike):
    """
    Selects the majority label. Useful for a classifier.

    Inputs
    -------
    labels: ArrayLike

    Output
    -------
    The most frequently occurring label in labels.

    Constraints
    -------
    labels must be ArrayLike and 1D.

    Examples
    -------
    >>> import numpy as np
    >>> labels = np.array(['a', 'b', 'b', 'c'])
    >>> print(majority_label(labels))
    >>> 'b'

    """
    labels = np.asarray(labels)
    _check_finite_if_numeric(labels)
    _check_consistent_types(labels)
    counts = {}

    for label in labels:
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1

    max_count = max(counts.values())

    candidates = [label for label, c in counts.items() if c == max_count]

    return min(candidates)

def average_label(labels: ArrayLike, dists: np.ndarray):
    """
    Selects the majority label. Useful for a classifier.

    Inputs
    -------
    labels: ArrayLike

    dists: ndarray of numeric values.

    Output
    -------
    The average label inversely weighted by dists.

    Constraints
    -------
    labels must be ArrayLike and 1D.

    Examples
    -------
    >>> import numpy as np
    >>> labels = np.array([1, 2, 3, 4])
    >>> dists = np.array([0, 1, 2, 1])
    >>> print(average_label(labels, dists))
    >>> 3

    """
    labels = np.asarray(labels)
    dists = np.asarray(dists)
    _check_numeric_dtypes(labels)
    _check_numeric_dtypes(dists)

    zero_mask = (dists == 0)
    if np.any(zero_mask):
        # If multiple zeros, average their labels
        return np.mean(labels[zero_mask])

    # CASE 2 — Distance weighted
    weights = 1.0 / dists
    return np.sum(labels * weights) / np.sum(weights)