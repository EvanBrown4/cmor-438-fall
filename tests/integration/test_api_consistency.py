import numpy as np

from rice_ml.supervised_learning.knn import KNNClassifier
from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.supervised_learning.perceptron import PerceptronClassifier
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
from rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier
from rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis
from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.unsupervised_learning.dbscan import DBSCAN


#------------------------------
## API Consistency Integration Tests
#------------------------------

#------------------------------
## fit() Should Return self
#------------------------------

def test_fit_returns_self_for_supervised_models():
    """All supervised models should return self from fit"""
    X = np.random.randn(50, 4)
    y_cls = np.random.randint(0, 2, size=50)
    y_reg = np.random.randn(50)

    models = [
        KNNClassifier(n_neighbors=3),
        LogisticRegression(),
        PerceptronClassifier(),
        MLPClassifier(hidden_layer_sizes=(8,)),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=5),
    ]

    for model in models:
        assert model.fit(X, y_cls) is model

    reg = DecisionTreeRegressor()
    assert reg.fit(X, y_reg) is reg

    lin = LinearRegression()
    assert lin.fit(X, y_reg) is lin


#------------------------------
## predict() Output Shape Consistency
#------------------------------

def test_predict_output_is_1d():
    """predict should return a 1D array of length n_samples"""
    X = np.random.randn(40, 3)
    y = np.random.randint(0, 2, size=40)

    models = [
        KNNClassifier(n_neighbors=3),
        LogisticRegression(),
        PerceptronClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=5),
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.ndim == 1
        assert preds.shape[0] == X.shape[0]


#------------------------------
## Regression predict() Shape
#------------------------------

def test_regression_predict_shape():
    """Regression models should output 1D continuous predictions"""
    X = np.random.randn(60, 2)
    y = np.random.randn(60)

    models = [
        LinearRegression(),
        DecisionTreeRegressor(),
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X)

        assert preds.ndim == 1
        assert preds.shape == y.shape


#------------------------------
## Unsupervised fit_predict() Shape
#------------------------------

def test_unsupervised_fit_predict_shape():
    """Unsupervised fit_predict should return 1D labels"""
    X = np.random.randn(80, 2)

    models = [
        KMeans(n_clusters=3),
        DBSCAN(eps=0.5),
    ]

    for model in models:
        labels = model.fit_predict(X)

        assert labels.ndim == 1
        assert labels.shape[0] == X.shape[0]


#------------------------------
## PCA API Consistency
#------------------------------

def test_pca_fit_transform_and_transform_shapes():
    """PCA should respect fit/transform API contract"""
    X = np.random.randn(100, 10)

    pca = PrincipalComponentAnalysis(n_components=4)
    X1 = pca.fit_transform(X)
    X2 = pca.transform(X)

    assert X1.shape == (100, 4)
    assert X2.shape == (100, 4)


#------------------------------
## Models Should Not Mutate Input
#------------------------------

def test_models_do_not_mutate_input():
    """Fitting models should not mutate input arrays"""
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, size=50)

    X_copy = X.copy()

    model = LogisticRegression()
    model.fit(X, y)

    assert np.allclose(X, X_copy)