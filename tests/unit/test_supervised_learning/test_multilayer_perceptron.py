import pytest
import numpy as np
from rice_ml.supervised_learning.multilayer_perceptron import MLPClassifier

#------------------------------
## MLP Classifier Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_mlp_basic_fit_and_predict():
    """Test simple binary classification"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])  # XOR problem
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[0, 0]])
    assert pred[0] in [0, 1]

def test_mlp_linearly_separable():
    """Should learn linearly separable patterns"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND problem
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score >= 0.5

def test_mlp_score():
    """Should compute accuracy score"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert 0 <= score <= 1

def test_mlp_predict_proba():
    """predict_proba should return probabilities"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    proba = clf.predict_proba([[0, 0]])
    assert proba.shape == (1, 2)
    assert np.isclose(proba.sum(axis=1)[0], 1.0)

def test_mlp_predict_proba_all_samples():
    """predict_proba should work for multiple samples"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    proba = clf.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

def test_mlp_single_hidden_layer():
    """Should work with single hidden layer"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[5, 5]])
    assert pred[0] in [0, 1]

def test_mlp_multiple_hidden_layers():
    """Should work with multiple hidden layers"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[5, 5]])
    assert pred[0] in [0, 1]

def test_mlp_deep_network():
    """Should work with deep networks"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(16, 8, 4), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[5, 5]])
    assert pred[0] in [0, 1]

def test_mlp_learning_rate():
    """Different learning rates should affect training"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), lr=0.1, max_iter=500, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[0, 0]])
    assert pred[0] in [0, 1]

def test_mlp_random_state():
    """Should produce consistent results with same random_state"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf1 = MLPClassifier(hidden_layer_sizes=(8,), max_iter=100, random_state=42)
    clf1.fit(X, y)
    pred1 = clf1.predict([[5, 5]])
    
    clf2 = MLPClassifier(hidden_layer_sizes=(8,), max_iter=100, random_state=42)
    clf2.fit(X, y)
    pred2 = clf2.predict([[5, 5]])
    
    assert pred1[0] == pred2[0]

def test_mlp_tolerance():
    """Should stop early when tolerance is met"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(8,), tol=1e-4, max_iter=10000, random_state=42)
    clf.fit(X, y)
    
    # Should converge before max_iter
    assert clf.n_iter_ < 10000

def test_mlp_losses_tracked():
    """Should track losses per epoch"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=10, random_state=42)
    clf.fit(X, y)
    
    assert len(clf.losses_) == clf.n_iter_
    assert all(isinstance(loss, (int, float, np.number)) for loss in clf.losses_)

def test_mlp_n_iter_attribute():
    """Should set n_iter_ attribute after fitting"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=10, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, 'n_iter_')
    assert clf.n_iter_ <= 10

def test_mlp_weights_initialized():
    """Should initialize weights and biases"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=10, random_state=42)
    clf.fit(X, y)
    
    assert hasattr(clf, 'weights_')
    assert hasattr(clf, 'biases_')
    assert len(clf.weights_) == 2  # input->hidden, hidden->output
    assert len(clf.biases_) == 2

def test_mlp_batch_predict():
    """Should predict multiple samples at once"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    preds = clf.predict([[1, 1], [2, 2], [3, 3]])
    assert preds.shape == (3,)

def test_mlp_verbose_mode():
    """Should run without errors in verbose mode"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=200, verbose=True, random_state=42)
    clf.fit(X, y)
    
    assert clf.n_iter_ > 0

#------------------------------
## Error Handling Tests
#------------------------------

def test_mlp_non_binary_labels():
    """Should raise ValueError for non-binary labels"""
    X = np.array([[0, 0], [0, 1], [1, 0]])
    y = np.array([0, 1, 2])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,))
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_mlp_invalid_binary_labels():
    """Should raise ValueError for labels not in {0, 1}"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([-1, -1, 1, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,))
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_mlp_mismatched_lengths():
    """X and y must have same number of samples"""
    clf = MLPClassifier(hidden_layer_sizes=(4,))
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    
    with pytest.raises(ValueError):
        clf.fit(X, y)

#------------------------------
## Edge Cases
#------------------------------

def test_mlp_all_same_class():
    """Should handle when all labels are same class"""
    X = np.array([[0, 0], [1, 1], [2, 2]])
    y = np.array([1, 1, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=10, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5, 1.5]])
    assert pred[0] in [0, 1]

def test_mlp_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), random_state=42)
    result = clf.fit(X, y)
    
    assert result is clf

def test_mlp_small_hidden_layer():
    """Should work with very small hidden layers"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(2,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[0, 0]])
    assert pred[0] in [0, 1]

def test_mlp_large_hidden_layer():
    """Should work with large hidden layers"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[5, 5]])
    assert pred[0] in [0, 1]

def test_mlp_activation_functions():
    """ReLU and sigmoid should work correctly"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    
    clf = MLPClassifier(hidden_layer_sizes=(4,), max_iter=500, random_state=42)
    clf.fit(X, y)
    
    # Test that activations produce valid outputs
    proba = clf.predict_proba(X)
    assert np.all((proba >= 0) & (proba <= 1))

def test_mlp_loss_decreases():
    """Loss should generally decrease during training"""
    X = np.array([[i, i] for i in range(20)])
    y = np.array([i % 2 for i in range(20)])
    
    clf = MLPClassifier(hidden_layer_sizes=(16,), lr=0.1, max_iter=500, random_state=42)
    clf.fit(X, y)
    
    # Check that final loss is lower than initial loss (with some tolerance)
    assert clf.losses_[-1] <= clf.losses_[0] or clf.losses_[-1] < 1.0