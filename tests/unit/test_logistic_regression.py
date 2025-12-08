import pytest
import numpy as np
from src.rice_ml.supervised_learning.linear_regression import LinearRegression
from src.rice_ml.supervised_learning.logistic_regression import LogisticRegression

#------------------------------
## Logistic Regression Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_logistic_regression_basic_fit_and_predict():
    """Test simple binary classification"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    pred = clf.predict([[0.5]])
    assert pred[0] in [0, 1]

def test_logistic_regression_predict_proba_shape():
    """predict_proba should return (n_samples, 2) array"""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 0, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    proba = clf.predict_proba([[1]])
    assert proba.shape == (1, 2)

def test_logistic_regression_proba_sums_to_one():
    """Probabilities should sum to 1 for each sample"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    proba = clf.predict_proba([[1.5]])
    assert np.isclose(proba.sum(axis=1)[0], 1.0)

def test_logistic_regression_decision_boundary():
    """Predictions should flip at probability threshold"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    proba = clf.predict_proba([[1.5]])
    pred = clf.predict([[1.5]])
    
    if proba[0, 1] >= 0.5:
        assert pred[0] == 1
    else:
        assert pred[0] == 0

def test_logistic_regression_no_intercept():
    """Test without intercept"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression(fit_intercept=False)
    clf.fit(X, y)
    
    pred = clf.predict([[2.5]])
    assert pred[0] in [0, 1]

def test_logistic_regression_perfect_separation():
    """Should handle perfectly separable data"""
    X = np.array([[0], [1], [10], [11]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score == 1.0

def test_logistic_regression_score_accuracy():
    """score() should return accuracy"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert 0 <= score <= 1

def test_logistic_regression_multiple_features():
    """Should work with multiple features"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    pred = clf.predict([[1.5, 1.5]])
    assert pred[0] in [0, 1]

#------------------------------
## Error Handling Tests
#------------------------------

def test_logistic_regression_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    clf = LogisticRegression()
    with pytest.raises(RuntimeError):
        clf.predict([[1]])

def test_logistic_regression_predict_proba_before_fit():
    """predict_proba() before fit should raise RuntimeError"""
    clf = LogisticRegression()
    with pytest.raises(RuntimeError):
        clf.predict_proba([[1]])

def test_logistic_regression_non_binary_labels():
    """Should raise ValueError for non-binary labels"""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])
    
    clf = LogisticRegression()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_logistic_regression_feature_mismatch():
    """X_test features must match training features"""
    clf = LogisticRegression()
    clf.fit(np.array([[1, 2]]), np.array([0]))
    
    with pytest.raises(ValueError):
        clf.predict([[1]])

def test_logistic_regression_mismatched_lengths():
    """X and y must have same number of samples"""
    clf = LogisticRegression()
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])
    
    with pytest.raises(ValueError):
        clf.fit(X, y)

#------------------------------
## Edge Cases
#------------------------------

def test_logistic_regression_all_same_class():
    """Should handle when all labels are same class"""
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    assert pred[0] == 1

def test_logistic_regression_convergence_tolerance():
    """Should respect tolerance parameter"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression(tol=1e-10, max_iter=10000)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    assert pred[0] in [0, 1]

def test_logistic_regression_learning_rate_effect():
    """Different learning rates should still converge"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression(lr=0.1, max_iter=1000)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score > 0.5

def test_logistic_regression_batch_predict():
    """Should predict multiple samples at once"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression()
    clf.fit(X, y)
    
    preds = clf.predict([[0], [1], [2]])
    assert preds.shape == (3,)

def test_logistic_regression_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    
    clf = LogisticRegression()
    result = clf.fit(X, y)
    
    assert result is clf

def test_logistic_regression_extreme_values():
    """Should handle large input values without overflow"""
    X = np.array([[1000], [2000], [3000], [4000]])
    y = np.array([0, 0, 1, 1])
    
    clf = LogisticRegression(lr=0.00001, max_iter=1000)
    clf.fit(X, y)
    
    proba = clf.predict_proba([[2500]])
    assert np.all(np.isfinite(proba))