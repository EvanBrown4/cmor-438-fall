import pytest
import numpy as np
from rice_ml.supervised_learning.linear_regression import LinearRegression

#------------------------------
## Linear Regression Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_linear_regression_basic_fit_and_predict():
    """Test simple linear fit with intercept"""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    
    pred = lr.predict([[4]])
    assert np.isclose(pred[0], 8, atol=1e-10)

def test_linear_regression_no_intercept():
    """Test regression without intercept"""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    
    assert np.isclose(lr.intercept_, 0.0)
    pred = lr.predict([[4]])
    assert np.isclose(pred[0], 8, atol=1e-10)

def test_linear_regression_stores_n_features():
    """fit() should store n_features_in"""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    assert lr.n_features_in == 3

def test_linear_regression_perfect_fit():
    """Perfect linear relationship should give R²=1"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([3, 5, 7, 9])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    score = lr.score(X, y)
    assert np.isclose(score, 1.0, atol=1e-10)

def test_linear_regression_multiple_features():
    """Test with multiple input features"""
    X = np.array([[1, 2], [2, 3], [3, 4]])
    y = np.array([5, 8, 11])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    pred = lr.predict([[4, 5]])
    assert len(pred) == 1

def test_linear_regression_batch_predict():
    """predict() should handle multiple samples"""
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    preds = lr.predict([[1], [2], [3]])
    assert preds.shape == (3,)

def test_linear_regression_coef_shape():
    """coef_ should have shape matching n_features"""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([1, 2, 3])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    assert lr.coef_.shape == (3,)

def test_linear_regression_intercept_stored():
    """intercept_ should be a scalar"""
    X = np.array([[1], [2]])
    y = np.array([3, 5])
    
    lr = LinearRegression(fit_intercept=True)
    lr.fit(X, y)
    
    assert isinstance(lr.intercept_, (float, np.floating))

#------------------------------
## Error Handling Tests
#------------------------------

def test_linear_regression_predict_before_fit():
    """Calling predict before fit should raise RuntimeError"""
    lr = LinearRegression()
    with pytest.raises(RuntimeError):
        lr.predict([[1]])

def test_linear_regression_feature_mismatch():
    """X_test feature size must match training feature size"""
    lr = LinearRegression()
    lr.fit(np.array([[1, 2]]), np.array([1]))
    
    with pytest.raises(ValueError):
        lr.predict([[1]])

def test_linear_regression_mismatched_lengths():
    """X and y must have same number of samples"""
    lr = LinearRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2])
    
    with pytest.raises(ValueError):
        lr.fit(X, y)

def test_linear_regression_score_before_fit():
    """score() before fit should raise RuntimeError"""
    lr = LinearRegression()
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    
    with pytest.raises(RuntimeError):
        lr.score(X, y)

#------------------------------
## Edge Cases
#------------------------------

def test_linear_regression_zero_variance_y():
    """When y has zero variance, score should handle gracefully"""
    X = np.array([[1], [2], [3]])
    y = np.array([5, 5, 5])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    score = lr.score(X, y)
    assert score == 1.0

def test_linear_regression_negative_correlation():
    """Regression should handle negative slopes"""
    X = np.array([[1], [2], [3]])
    y = np.array([6, 4, 2])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    pred = lr.predict([[4]])
    assert pred[0] < 2

def test_linear_regression_single_sample():
    """Should fit with just one sample"""
    X = np.array([[1]])
    y = np.array([2])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    pred = lr.predict([[1]])
    assert len(pred) == 1

def test_linear_regression_large_values():
    """Should handle large input values"""
    X = np.array([[1000], [2000], [3000]])
    y = np.array([5000, 10000, 15000])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    pred = lr.predict([[4000]])
    assert np.isfinite(pred[0])

def test_linear_regression_negative_score():
    """Score can be negative for bad fits"""
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    # Test on very different data
    X_test = np.array([[1], [2], [3]])
    y_test = np.array([100, 200, 300])
    
    score = lr.score(X_test, y_test)
    assert score < 0

def test_linear_regression_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    
    lr = LinearRegression()
    result = lr.fit(X, y)
    
    assert result is lr

def test_linear_regression_float_output():
    """predict() should return float array"""
    X = np.array([[1], [2]])
    y = np.array([1, 2])
    
    lr = LinearRegression()
    lr.fit(X, y)
    
    pred = lr.predict([[1]])
    assert isinstance(pred[0], (float, np.floating))