"""


"""
import numpy as np

def train_test_split(x: np.array, y: np.array):
    pass


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