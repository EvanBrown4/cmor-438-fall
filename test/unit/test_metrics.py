import pytest
import numpy as np
import math
from src.utilities.metrics import euclidean_dist, manhattan_dist

# def test_euclid_basic():
#     x = np.array([0, 0])
#     y = np.array([0, 1])
#     assert euclidean_dist(x, y) == 1.0

# def test_euclid_zero_dist():
#     x = np.array([0, 0])
#     y = np.array([0, 0])
#     assert euclidean_dist(x, y) == 0.0

# def test_euclid_negative_points():
#     x = np.array([-1, -3])
#     y = np.array([-4, -2])
#     assert euclidean_dist(x, y) == math.sqrt(8)

# def test_euclid_multi_dims():
#     pass

# def test_manhattan_basic():
#     x = np.array([0, 0])
#     y = np.array([0, 1])
#     assert manhattan_dist(x, y) == 1.0

# def test_manhattan_zero_dist():
#     x = np.array([0, 0])
#     y = np.array([0, 0])
#     assert manhattan_dist(x, y) == 0.0

# def test_manhattan_negative_points():
#     pass

# def test_manhattan_multi_dims():
#     pass