"""
knn() trains a knn model on the values given.
"""
from .metrics import euclidean_dist, manhattan_dist
from .preprocess import train_test_split, normalize, scale
import numpy as np

def knn(x: np.array, y: np.array, dist: "euclidean"):
    if len(x) != len(y):
        # ERROR! TODO: HANDLE
        return
    pass
    x_train, y_train, x_test, y_test = train_test_split(x, y)
