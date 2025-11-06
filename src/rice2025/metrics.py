"""
metrics.py
Contains any necessary metric computations.

euclidean_dist: Computes the euclidean distance (l2) of two numpy arrays.
manhattan_dist: Computes the manahattan distance (l1) of two numpy arrays.
"""

import math
import numpy as np

def euclidean_dist(x: np.array, y: np.array):
    """
    Computes the euclidean distance (l2) of the arrays X and Y.
    """
    if len(x) != len(y):
        # ERROR! TODO: HANDLE
        return -1
    return math.sqrt(sum((x-y)**2))

def manhattan_dist(x: np.array, y: np.array):
    """
    Computes the manhattan distance (l1) of the arrays X and Y.
    """
    if len(x) != len(y):
        # ERROR! TODO: HANDLE
        return -1
    return sum(math.abs(x-y))