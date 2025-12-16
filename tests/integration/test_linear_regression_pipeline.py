import numpy as np

from rice_ml.supervised_learning.linear_regression import LinearRegression
from rice_ml.utilities.metrics import r2_score


#------------------------------
## Linear Regression Integration Tests
#------------------------------

def test_linear_regression_end_to_end():
    """Test regression learns linear relationship"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3))
    y = 4 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.1, size=300)

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    r2 = r2_score(y, preds)

    assert r2 > 0.9


def test_linear_regression_prediction_shape():
    """Prediction shape should match input length"""
    X = np.random.randn(50, 2)
    y = np.random.randn(50)

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == y.shape
