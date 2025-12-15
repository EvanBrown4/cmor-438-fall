import pytest
import numpy as np
from rice_ml.supervised_learning.perceptron import PerceptronClassifier

#------------------------------
## Perceptron Classifier Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_perceptron_basic_fit_and_predict():
    """Test simple binary classification"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(max_iter=100)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    assert pred[0] in [0, 1]

def test_perceptron_linearly_separable():
    """Should perfectly classify linearly separable data"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = PerceptronClassifier(max_iter=1000)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score == 1.0

def test_perceptron_score():
    """Should compute accuracy score"""
    X = np.array([[i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = PerceptronClassifier(max_iter=100)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert 0 <= score <= 1

def test_perceptron_multiple_features():
    """Should work with multiple features"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(max_iter=100)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5, 1.5]])
    assert pred[0] in [0, 1]

def test_perceptron_no_intercept():
    """Test without intercept"""
    X = np.array([[1], [2], [3], [4]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(fit_intercept=False, max_iter=100)
    clf.fit(X, y)
    
    pred = clf.predict([[2.5]])
    assert pred[0] in [0, 1]

def test_perceptron_with_intercept():
    """Test with intercept (default behavior)"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(fit_intercept=True, max_iter=100)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    assert pred[0] in [0, 1]

def test_perceptron_shuffle():
    """Test with shuffle enabled"""
    np.random.seed(42)
    X = np.array([[i] for i in range(20)])
    y = np.array([i % 2 for i in range(20)])
    
    clf = PerceptronClassifier(shuffle=True, max_iter=100)
    clf.fit(X, y)
    
    pred = clf.predict([[5]])
    assert pred[0] in [0, 1]

def test_perceptron_no_shuffle():
    """Test with shuffle disabled"""
    X = np.array([[i] for i in range(20)])
    y = np.array([i % 2 for i in range(20)])
    
    clf = PerceptronClassifier(shuffle=False, max_iter=100)
    clf.fit(X, y)
    
    pred = clf.predict([[5]])
    assert pred[0] in [0, 1]

def test_perceptron_learning_rate():
    """Different learning rates should still converge"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(lr=0.1, max_iter=100)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score > 0.5

def test_perceptron_tolerance():
    """Should stop early when tolerance is met"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = PerceptronClassifier(tol=0, max_iter=1000)
    clf.fit(X, y)
    
    # Should converge before max_iter
    assert clf.n_iter_ < 1000

def test_perceptron_errors_tracked():
    """Should track errors per epoch"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(max_iter=10)
    clf.fit(X, y)
    
    assert len(clf.errors_) == clf.n_iter_
    assert all(isinstance(e, int) for e in clf.errors_)

def test_perceptron_n_iter_attribute():
    """Should set n_iter_ attribute after fitting"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(max_iter=10)
    clf.fit(X, y)
    
    assert hasattr(clf, 'n_iter_')
    assert clf.n_iter_ <= 10

def test_perceptron_coef_shape():
    """Coefficients should have correct shape"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(fit_intercept=True, max_iter=100)
    clf.fit(X, y)
    
    # With intercept: n_features + 1
    assert clf.coef_.shape[0] == 3

def test_perceptron_batch_predict():
    """Should predict multiple samples at once"""
    X = np.array([[i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = PerceptronClassifier(max_iter=100)
    clf.fit(X, y)
    
    preds = clf.predict([[1], [2], [3]])
    assert preds.shape == (3,)

#------------------------------
## Error Handling Tests
#------------------------------

def test_perceptron_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    clf = PerceptronClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[1]])

def test_perceptron_non_binary_labels():
    """Should raise ValueError for non-binary labels"""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])
    
    clf = PerceptronClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_perceptron_invalid_binary_labels():
    """Should raise ValueError for labels not in {0, 1}"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([-1, -1, 1, 1])
    
    clf = PerceptronClassifier()
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_perceptron_feature_mismatch():
    """X_test features must match training features"""
    clf = PerceptronClassifier()
    clf.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    
    with pytest.raises(ValueError):
        clf.predict([[1]])

def test_perceptron_mismatched_lengths():
    """X and y must have same number of samples"""
    clf = PerceptronClassifier()
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])
    
    with pytest.raises(ValueError):
        clf.fit(X, y)

#------------------------------
## Edge Cases
#------------------------------

def test_perceptron_all_same_class():
    """Should handle when all labels are same class"""
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])
    
    clf = PerceptronClassifier(max_iter=10)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    assert pred[0] == 1

def test_perceptron_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    
    clf = PerceptronClassifier()
    result = clf.fit(X, y)
    
    assert result is clf

def test_perceptron_zero_tolerance():
    """Should work with zero tolerance"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = PerceptronClassifier(tol=0, max_iter=1000)
    clf.fit(X, y)
    
    # Should stop when no errors
    assert clf.errors_[-1] == 0

def test_perceptron_single_sample():
    """Should handle training on single sample"""
    X = np.array([[1, 2]])
    y = np.array([0])
    
    clf = PerceptronClassifier(max_iter=10)
    clf.fit(X, y)
    
    pred = clf.predict([[1, 2]])
    assert pred[0] in [0, 1]

def test_perceptron_activation_function():
    """Activation should return binary values"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = PerceptronClassifier(max_iter=100)
    clf.fit(X, y)
    
    preds = clf.predict([[0], [1], [2], [3]])
    assert all(p in [0, 1] for p in preds)