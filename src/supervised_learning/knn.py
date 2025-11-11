import numpy as np
from src.utilities import *

def knn(x, y, dist: "euclidean"):
    """
    x and y expected: numpy array
    """
    if len(x) != len(y):
        # ERROR! TODO: HANDLE
        return
    x_train, y_train, x_test, y_test = train_test_split(x, y)
