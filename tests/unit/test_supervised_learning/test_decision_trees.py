import pytest
import numpy as np
from rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

#------------------------------
## Decision Tree Classifier Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_classifier_basic_fit_and_predict():
    """Test simple binary classification"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    pred = clf.predict([[0.5]])
    assert pred[0] in [0, 1]

def test_classifier_perfect_fit():
    """Should achieve perfect accuracy on training data"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score == 1.0

def test_classifier_multiclass():
    """Should handle multiple classes"""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 2, 2])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    pred = clf.predict([[1], [3], [5]])
    assert len(pred) == 3
    assert all(p in [0, 1, 2] for p in pred)

def test_classifier_multiple_features():
    """Should work with multiple features"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    pred = clf.predict([[1.5, 1.5]])
    assert pred[0] in [0, 1]

def test_classifier_max_depth():
    """Should respect max_depth parameter"""
    X = np.array([[i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    
    # Different definitions of depth:
    #   max_depth is max number of branches/splits allowed
    #   get_depth returns number of nodes on the longest root to leaf path
    depth = clf.get_depth()
    assert depth <= 3 

def test_classifier_min_samples_split():
    """Should respect min_samples_split parameter"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier(min_samples_split=3)
    clf.fit(X, y)
    
    # Tree should still be built but with restrictions
    pred = clf.predict([[1.5]])
    assert pred[0] in [0, 1]

def test_classifier_min_samples_leaf():
    """Should respect min_samples_leaf parameter"""
    X = np.array([[i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = DecisionTreeClassifier(min_samples_leaf=2)
    clf.fit(X, y)
    
    pred = clf.predict([[5]])
    assert pred[0] in [0, 1]

def test_classifier_entropy_criterion():
    """Should work with entropy criterion"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score == 1.0

def test_classifier_gini_criterion():
    """Should work with gini criterion"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert score == 1.0

def test_classifier_get_n_leaves():
    """Should correctly count leaf nodes"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(X, y)
    
    n_leaves = clf.get_n_leaves()
    assert n_leaves >= 1

#------------------------------
## Decision Tree Regressor Tests
#------------------------------

def test_regressor_basic_fit_and_predict():
    """Test simple regression"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    pred = reg.predict([[1.5]])
    assert isinstance(pred[0], (int, float, np.number))

def test_regressor_perfect_fit():
    """Should achieve perfect R^2 on training data"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    score = reg.score(X, y)
    assert score == 1.0

def test_regressor_multiple_features():
    """Should work with multiple features"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0.0, 2.0, 4.0, 6.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    pred = reg.predict([[1.5, 1.5]])
    assert isinstance(pred[0], (int, float, np.number))

def test_regressor_max_depth():
    """Should respect max_depth parameter"""
    X = np.array([[i] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = DecisionTreeRegressor(max_depth=3)
    reg.fit(X, y)

    # Different definitions of depth:
    #   max_depth is max number of branches/splits allowed
    #   get_depth returns number of nodes on the longest root to leaf path
    depth = reg.get_depth()
    assert depth <= 4

#------------------------------
## Error Handling Tests
#------------------------------

def test_classifier_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[1]])

def test_regressor_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    reg = DecisionTreeRegressor()
    with pytest.raises(RuntimeError):
        reg.predict([[1]])

def test_classifier_feature_mismatch():
    """X_test features must match training features"""
    clf = DecisionTreeClassifier()
    clf.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    
    with pytest.raises(ValueError):
        clf.predict([[1]])

def test_classifier_invalid_criterion():
    """Should raise ValueError for invalid criterion"""
    with pytest.raises(ValueError):
        clf = DecisionTreeClassifier(criterion='invalid') # type: ignore

def test_classifier_mismatched_lengths():
    """X and y must have same number of samples"""
    clf = DecisionTreeClassifier()
    X = np.array([[1], [2], [3]])
    y = np.array([0, 1])
    
    with pytest.raises(ValueError):
        clf.fit(X, y)

def test_classifier_get_depth_before_fit():
    """get_depth() before fit should raise RuntimeError"""
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.get_depth()

def test_classifier_get_n_leaves_before_fit():
    """get_n_leaves() before fit should raise RuntimeError"""
    clf = DecisionTreeClassifier()
    with pytest.raises(RuntimeError):
        clf.get_n_leaves()

#------------------------------
## Edge Cases
#------------------------------

def test_classifier_single_class():
    """Should handle when all labels are same class"""
    X = np.array([[0], [1], [2]])
    y = np.array([1, 1, 1])
    
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    
    pred = clf.predict([[1.5]])
    assert pred[0] == 1

def test_classifier_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0], [1]])
    y = np.array([0, 1])
    
    clf = DecisionTreeClassifier()
    result = clf.fit(X, y)
    
    assert result is clf

def test_regressor_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0], [1]])
    y = np.array([0.0, 1.0])
    
    reg = DecisionTreeRegressor()
    result = reg.fit(X, y)
    
    assert result is reg