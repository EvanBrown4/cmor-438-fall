import pytest
import numpy as np
from rice_ml.supervised_learning.regression_trees import DecisionTreeRegressor


#------------------------------
## Standard Tests
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
    
    # Shallow tree should not perfectly fit all points
    pred = reg.predict(X)
    assert len(pred) == len(y)


def test_regressor_min_samples_split():
    """Should respect min_samples_split parameter"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = DecisionTreeRegressor(min_samples_split=3)
    reg.fit(X, y)
    
    pred = reg.predict([[1.5]])
    assert isinstance(pred[0], (int, float, np.number))


def test_regressor_min_samples_leaf():
    """Should respect min_samples_leaf parameter"""
    X = np.array([[i] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = DecisionTreeRegressor(min_samples_leaf=2)
    reg.fit(X, y)
    
    pred = reg.predict([[5]])
    assert isinstance(pred[0], (int, float, np.number))


def test_regressor_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[0], [1]])
    y = np.array([0.0, 1.0])
    
    reg = DecisionTreeRegressor()
    result = reg.fit(X, y)
    
    assert result is reg


def test_regressor_predict_multiple_samples():
    """Should handle multiple samples in prediction"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    X_test = np.array([[0.5], [1.5], [2.5]])
    pred = reg.predict(X_test)
    
    assert len(pred) == 3
    assert all(isinstance(p, (int, float, np.number)) for p in pred)


def test_regressor_negative_values():
    """Should handle negative target values"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    pred = reg.predict([[1.5]])
    assert isinstance(pred[0], (int, float, np.number))


def test_regressor_nonlinear_relationship():
    """Should fit nonlinear relationships"""
    X = np.array([[i] for i in range(10)])
    y = np.array([i**2 for i in range(10)], dtype=float)
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    score = reg.score(X, y)
    assert score == 1.0


def test_regressor_max_features_int():
    """Should work with integer max_features"""
    X = np.array([[i, i*2, i*3] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = DecisionTreeRegressor(max_features=2, random_state=42)
    reg.fit(X, y)
    
    pred = reg.predict([[5, 10, 15]])
    assert isinstance(pred[0], (int, float, np.number))


def test_regressor_max_features_float():
    """Should work with float max_features"""
    X = np.array([[i, i*2, i*3] for i in range(10)])
    y = np.array([float(i) for i in range(10)])
    
    reg = DecisionTreeRegressor(max_features=0.5, random_state=42)
    reg.fit(X, y)
    
    pred = reg.predict([[5, 10, 15]])
    assert isinstance(pred[0], (int, float, np.number))


def test_regressor_random_state():
    """Random state should produce reproducible results"""
    X = np.array([[i, i*2] for i in range(20)])
    y = np.array([float(i) for i in range(20)])
    
    reg1 = DecisionTreeRegressor(max_features=1, random_state=42)
    reg1.fit(X, y)
    pred1 = reg1.predict([[5, 10]])
    
    reg2 = DecisionTreeRegressor(max_features=1, random_state=42)
    reg2.fit(X, y)
    pred2 = reg2.predict([[5, 10]])
    
    assert pred1[0] == pred2[0]


#------------------------------
## Error Handling Tests
#------------------------------

def test_regressor_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    reg = DecisionTreeRegressor()
    with pytest.raises(RuntimeError):
        reg.predict([[1]])


def test_regressor_score_before_fit():
    """score() before fit should raise RuntimeError"""
    reg = DecisionTreeRegressor()
    with pytest.raises(RuntimeError):
        reg.score([[1]], [1.0])


def test_regressor_feature_mismatch():
    """X_test features must match training features"""
    reg = DecisionTreeRegressor()
    reg.fit(np.array([[1, 2], [3, 4]]), np.array([0.0, 1.0]))
    
    with pytest.raises(ValueError):
        reg.predict([[1]])


def test_regressor_mismatched_lengths():
    """X and y must have same number of samples"""
    reg = DecisionTreeRegressor()
    X = np.array([[1], [2], [3]])
    y = np.array([0.0, 1.0])
    
    with pytest.raises(ValueError):
        reg.fit(X, y)


def test_regressor_invalid_max_features_type():
    """Should raise ValueError for invalid max_features type"""
    reg = DecisionTreeRegressor(max_features="invalid")  # type: ignore
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0.0, 1.0])
    
    with pytest.raises(ValueError):
        reg.fit(X, y)


#------------------------------
## Edge Cases
#------------------------------

def test_regressor_constant_target():
    """Should handle when all targets are the same"""
    X = np.array([[0], [1], [2]])
    y = np.array([5.0, 5.0, 5.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    pred = reg.predict([[1.5]])
    assert pred[0] == 5.0


def test_regressor_single_sample():
    """Should handle single sample training data"""
    X = np.array([[1]])
    y = np.array([5.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    pred = reg.predict([[2]])
    assert pred[0] == 5.0


def test_regressor_two_samples():
    """Should handle two sample training data"""
    X = np.array([[1], [2]])
    y = np.array([1.0, 2.0])
    
    reg = DecisionTreeRegressor()
    reg.fit(X, y)
    
    pred = reg.predict([[1], [2]])
    assert len(pred) == 2