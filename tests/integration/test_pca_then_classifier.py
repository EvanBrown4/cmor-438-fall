import pytest
import numpy as np

from rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis
from rice_ml.supervised_learning.knn import KNNClassifier
from rice_ml.supervised_learning.logistic_regression import LogisticRegression
from rice_ml.supervised_learning.perceptron import PerceptronClassifier
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier
from rice_ml.utilities import normalize


#------------------------------
## PCA + Classifier Integration Tests
#------------------------------

@pytest.mark.parametrize("model", [
    KNNClassifier(n_neighbors=5),
    LogisticRegression(),
    PerceptronClassifier(lr=0.1, max_iter=3000, random_state=42),
    MLPClassifier(hidden_layer_sizes=(16,), lr=0.05, max_iter=3000, random_state=42)
])

def test_pca_then_classifier_pipeline(model):
    rng = np.random.default_rng(42)

    X = rng.normal(size=(300, 20))

    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    X = normalize(X, method="zscore")

    pca = PrincipalComponentAnalysis(n_components=5)
    X_pca = pca.fit_transform(X)

    model.fit(X_pca, y)
    preds = model.predict(X_pca)

    assert preds.shape == y.shape
    acc = np.mean(preds == y)
    assert acc > 0.55



def test_pca_fit_transform_vs_transform_consistency():
    """fit_transform and transform should give same result"""
    X = np.random.randn(200, 15)

    pca = PrincipalComponentAnalysis(n_components=6)
    X1 = pca.fit_transform(X)
    X2 = pca.transform(X)

    assert X1.shape == X2.shape
    assert np.allclose(X1, X2)
