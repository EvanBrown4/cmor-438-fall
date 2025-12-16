import numpy as np

from rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis


#------------------------------
## PCA Component Integration Tests
#------------------------------

def test_pca_dimension_reduction():
    """PCA should reduce dimensionality"""
    X = np.random.randn(150, 20)

    pca = PrincipalComponentAnalysis(n_components=5)
    X_pca = pca.fit_transform(X)

    assert X_pca.shape == (150, 5)
    assert pca.n_features_in_ == 20


def test_pca_explained_variance_monotonic():
    """Cumulative explained variance should increase"""
    X = np.random.randn(200, 30)

    pca = PrincipalComponentAnalysis(n_components=10)
    pca.fit(X)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    assert np.all(np.diff(cumvar) >= 0)