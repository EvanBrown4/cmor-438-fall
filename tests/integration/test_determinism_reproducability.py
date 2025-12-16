import numpy as np

from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis


#------------------------------
## Determinism / Reproducibility Integration Tests
#------------------------------

def test_random_forest_reproducibility():
    """Random forest should be deterministic given same random_state"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)

    rf1 = RandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        random_state=42,
    )
    rf2 = RandomForestClassifier(
        n_estimators=10,
        max_depth=4,
        random_state=42,
    )

    rf1.fit(X, y)
    rf2.fit(X, y)

    assert np.all(rf1.predict(X) == rf2.predict(X))


def test_mlp_reproducibility():
    """MLP should produce same predictions with same random_state"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    mlp1 = MLPClassifier(
        hidden_layer_sizes=(16,),
        random_state=123,
        max_iter=200,
    )
    mlp2 = MLPClassifier(
        hidden_layer_sizes=(16,),
        random_state=123,
        max_iter=200,
    )

    mlp1.fit(X, y)
    mlp2.fit(X, y)

    assert np.all(mlp1.predict(X) == mlp2.predict(X))


def test_kmeans_reproducibility():
    """KMeans clustering should be deterministic with fixed random_state"""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(100, 2))

    km1 = KMeans(n_clusters=3, random_state=7)
    km2 = KMeans(n_clusters=3, random_state=7)

    labels1 = km1.fit_predict(X)
    labels2 = km2.fit_predict(X)

    assert np.all(labels1 == labels2)


def test_pca_reproducibility():
    """PCA should be deterministic for fixed input"""
    X = np.random.randn(120, 8)

    pca1 = PrincipalComponentAnalysis(n_components=4)
    pca2 = PrincipalComponentAnalysis(n_components=4)

    X1 = pca1.fit_transform(X)
    X2 = pca2.fit_transform(X)

    assert np.allclose(X1, X2)
