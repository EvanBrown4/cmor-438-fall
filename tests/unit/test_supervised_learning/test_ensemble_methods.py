import pytest
import numpy as np
from src.rice_ml.supervised_learning.ensemble_methods import (
    RandomForestClassifier,
    RandomForestRegressor,
)

#------------------------------
## Random Forest Classifier Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_rf_classifier_basic_fit_and_predict():
    """Test simple binary classification"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5, 1.5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_multiclass():
    """Should handle multiple classes"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    y = np.array([0, 0, 1, 1, 2, 2])
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[1, 1], [3, 3], [5, 5]])
    assert len(pred) == 3
    assert all(p in [0, 1, 2] for p in pred)

def test_rf_classifier_multiple_estimators():
    """Should create specified number of trees"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    
    assert len(clf.trees_) == 5
    assert len(clf._feature_indices_) == 5

def test_rf_classifier_max_depth():
    """Should respect max_depth parameter"""
    X = np.array([[i, i] for i in range(20)])
    y = np.array([i % 2 for i in range(20)])
    
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[5, 5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_max_features_sqrt():
    """Should work with sqrt max_features"""
    X = np.array([[i, i, i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = RandomForestClassifier(n_estimators=5, max_features='sqrt', random_state=42)
    clf.fit(X, y)
    
    assert clf._n_features_per_tree == 2  # sqrt(4) = 2
    pred = clf.predict([[5, 5, 5, 5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_max_features_log2():
    """Should work with log2 max_features"""
    X = np.array([[i, i, i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = RandomForestClassifier(n_estimators=5, max_features='log2', random_state=42)
    clf.fit(X, y)
    
    assert clf._n_features_per_tree == 2  # log2(4) = 2
    pred = clf.predict([[5, 5, 5, 5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_max_features_int():
    """Should work with integer max_features"""
    X = np.array([[i, i, i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = RandomForestClassifier(n_estimators=5, max_features=2, random_state=42)
    clf.fit(X, y)
    
    assert clf._n_features_per_tree == 2
    pred = clf.predict([[5, 5, 5, 5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_max_features_all():
    """Should work with 'all' max_features"""
    X = np.array([[i, i, i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = RandomForestClassifier(n_estimators=5, max_features='all', random_state=42)
    clf.fit(X, y)
    
    assert clf._n_features_per_tree == 4
    pred = clf.predict([[5, 5, 5, 5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_no_bootstrap():
    """Should work without bootstrap sampling"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])
    
    clf = RandomForestClassifier(n_estimators=5, bootstrap=False, random_state=42)
    clf.fit(X, y)
    
    pred = clf.predict([[1.5, 1.5]])
    assert pred[0] in [0, 1]

def test_rf_classifier_random_state():
    """Should produce consistent results with same random_state"""
    X = np.array([[i, i] for i in range(20)])
    y = np.array([i % 2 for i in range(20)])
    
    clf1 = RandomForestClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)
    pred1 = clf1.predict([[5, 5]])
    
    clf2 = RandomForestClassifier(n_estimators=10, random_state=42)
    clf2.fit(X, y)
    pred2 = clf2.predict([[5, 5]])
    
    assert pred1[0] == pred2[0]

def test_rf_classifier_score():
    """Should compute accuracy score"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    
    score = clf.score(X, y)
    assert 0 <= score <= 1

def test_rf_classifier_batch_predict():
    """Should predict multiple samples at once"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([i % 2 for i in range(10)])
    
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    
    preds = clf.predict([[1, 1], [2, 2], [3, 3]])
    assert preds.shape == (3,)

def test_rf_classifier_repr():
    """Should have a readable string representation"""
    clf = RandomForestClassifier(n_estimators=5, max_depth=3, max_features='sqrt')
    repr_str = repr(clf)
    
    assert 'RandomForestClassifier' in repr_str
    assert 'n_estimators=5' in repr_str
    assert 'max_depth=3' in repr_str

#------------------------------
## Random Forest Regressor Tests
#------------------------------

def test_rf_regressor_basic_fit_and_predict():
    """Test simple regression"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y)
    
    pred = reg.predict([[1.5, 1.5]])
    assert isinstance(pred[0], (int, float, np.number))

def test_rf_regressor_score():
    """Should compute R^2 score"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = RandomForestRegressor(n_estimators=10, random_state=42)
    reg.fit(X, y)
    
    score = reg.score(X, y)
    assert 0 <= score <= 1

def test_rf_regressor_multiple_estimators():
    """Should create specified number of trees"""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = RandomForestRegressor(n_estimators=7, random_state=42)
    reg.fit(X, y)
    
    assert len(reg.trees_) == 7
    assert len(reg._feature_indices_) == 7

def test_rf_regressor_max_features():
    """Should work with different max_features settings"""
    X = np.array([[i, i, i, i] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = RandomForestRegressor(n_estimators=5, max_features='sqrt', random_state=42)
    reg.fit(X, y)
    
    assert reg._n_features_per_tree == 2
    pred = reg.predict([[5, 5, 5, 5]])
    assert isinstance(pred[0], (int, float, np.number))

def test_rf_regressor_batch_predict():
    """Should predict multiple samples at once"""
    X = np.array([[i, i] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = RandomForestRegressor(n_estimators=5, random_state=42)
    reg.fit(X, y)
    
    preds = reg.predict([[1, 1], [2, 2], [3, 3]])
    assert preds.shape == (3,)

def test_rf_regressor_repr():
    """Should have a readable string representation"""
    reg = RandomForestRegressor(n_estimators=5, max_depth=3, max_features='log2')
    repr_str = repr(reg)
    
    assert 'RandomForestRegressor' in repr_str
    assert 'n_estimators=5' in repr_str
    assert 'max_depth=3' in repr_str

#------------------------------
## Error Handling Tests
#------------------------------

def test_rf_classifier_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    clf = RandomForestClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[1, 1]])

def test_rf_regressor_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    reg = RandomForestRegressor()
    with pytest.raises(RuntimeError):
        reg.predict([[1, 1]])

def test_rf_classifier_feature_mismatch():
    """X_test features must match training features"""
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(np.array([[1, 2], [3, 4]]), np.array([0, 1]))
    
    with pytest.raises(ValueError):
        clf.predict([[1]])

def test_rf_classifier_invalid_n_estimators():
    """Should raise ValueError for invalid n_estimators"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(n_estimators=0)
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_max_depth():
    """Should raise ValueError for invalid max_depth"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(max_depth=0)
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_min_samples_split():
    """Should raise ValueError for invalid min_samples_split"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(min_samples_split=1)
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_min_samples_leaf():
    """Should raise ValueError for invalid min_samples_leaf"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(min_samples_leaf=0)
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_max_features_string():
    """Should raise ValueError for invalid max_features string"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(max_features='invalid') # type: ignore
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_max_features_int():
    """Should raise ValueError for max_features > n_features"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(max_features=10)
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_max_features_zero():
    """Should raise ValueError for max_features = 0"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(max_features=0)
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_invalid_bootstrap():
    """Should raise ValueError for non-boolean bootstrap"""
    with pytest.raises(ValueError):
        clf = RandomForestClassifier(bootstrap='yes') # type: ignore
        clf.fit(np.array([[1, 2]]), np.array([0]))

def test_rf_classifier_mismatched_lengths():
    """X and y must have same number of samples"""
    clf = RandomForestClassifier()
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])
    
    with pytest.raises(ValueError):
        clf.fit(X, y)

#------------------------------
## Edge Cases
#------------------------------

def test_rf_classifier_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    result = clf.fit(X, y)
    
    assert result is clf

def test_rf_regressor_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0.0, 1.0])
    
    reg = RandomForestRegressor(n_estimators=5, random_state=42)
    result = reg.fit(X, y)
    
    assert result is reg

def test_rf_classifier_clears_previous_trees():
    """Should clear trees from previous fit"""
    X = np.array([[0, 0], [1, 1]])
    y = np.array([0, 1])
    
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    first_tree_count = len(clf.trees_)
    
    clf.fit(X, y)
    second_tree_count = len(clf.trees_)
    
    assert first_tree_count == second_tree_count == 5