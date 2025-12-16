import pytest
import numpy as np

from rice_ml.supervised_learning.knn import KNNClassifier
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis
from rice_ml.unsupervised_learning.k_means_clustering import KMeans


#------------------------------
## Error Path Integration Tests
#------------------------------

#------------------------------
## Predict Before Fit
#------------------------------

@pytest.mark.parametrize("model", [
    KNNClassifier(n_neighbors=3),
    LogisticRegression(),
    LinearRegression(),
    DecisionTreeClassifier(),
    DecisionTreeRegressor(),
])
def test_predict_before_fit_raises(model):
    """Calling predict before fit should raise a runtime error"""
    X = np.random.randn(10, 3)

    with pytest.raises(RuntimeError):
        model.predict(X)


#------------------------------
## Transform Before Fit
#------------------------------

def test_pca_transform_before_fit_raises():
    """PCA.transform before fit should fail"""
    X = np.random.randn(10, 5)
    pca = PrincipalComponentAnalysis(n_components=3)

    with pytest.raises(RuntimeError):
        pca.transform(X)


#------------------------------
## Mismatched Feature Dimensions
#------------------------------

def test_predict_with_wrong_feature_dim_raises():
    """Predicting with wrong feature dimension should fail"""
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, size=50)

    clf = LogisticRegression()
    clf.fit(X, y)

    X_bad = np.random.randn(10, 3)
    with pytest.raises(ValueError):
        clf.predict(X_bad)


#------------------------------
## PCA + Model Dimension Mismatch
#------------------------------

def test_pipeline_dimension_mismatch_raises():
    """Passing non-PCA-transformed data to PCA-trained model should fail"""
    X = np.random.randn(100, 10)
    y = (X[:, 0] > 0).astype(int)

    pca = PrincipalComponentAnalysis(n_components=4)
    X_pca = pca.fit_transform(X)

    clf = LogisticRegression()
    clf.fit(X_pca, y)

    with pytest.raises(ValueError):
        clf.predict(X)  # wrong dimensionality


#------------------------------
## Unsupervised Predict Before Fit
#------------------------------

def test_kmeans_predict_before_fit_raises():
    """KMeans predict before fit should fail"""
    km = KMeans(n_clusters=3)
    X = np.random.randn(20, 2)

    with pytest.raises(RuntimeError):
        km.predict(X)
