import pytest
import numpy as np
from src.rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor

#------------------------------
## KNN Classifier Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_knn_classifier_basic_fit_and_predict():
    """Test simple classification with k=1"""
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    clf = KNNClassifier(n_neighbors=1)
    clf.fit(X, y)

    pred = clf.predict([[0.2]])
    assert pred[0] == 0

def test_knn_classifier_two_neighbors_uniform():
    """Test classifier with k=2 uniform weights"""
    X = np.array([[0], [10]])
    y = np.array([1, 2])

    clf = KNNClassifier(n_neighbors=2, weights="uniform")
    clf.fit(X, y)

    pred = clf.predict([[3]])
    assert pred[0] in [1, 2]

def test_knn_classifier_predict_proba_rowsum():
    """Test that probability rows sum to 1"""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 1])

    clf = KNNClassifier(n_neighbors=2)
    clf.fit(X, y)

    proba = clf.predict_proba([[1]])
    assert np.isclose(proba.sum(), 1.0)

def test_knn_classifier_distance_weighting_prefers_closer():
    """Distance weighting must prefer the closest neighbor"""
    X = np.array([[0], [100]])
    y = np.array([1, 2])

    clf = KNNClassifier(n_neighbors=2, weights="distance")
    clf.fit(X, y)

    pred = clf.predict([[1]])
    assert pred[0] == 1

#------------------------------
## Error Handling Tests
#------------------------------

def test_knn_classifier_predict_before_fit():
    """Calling predict before fit should raise"""
    clf = KNNClassifier()
    with pytest.raises(RuntimeError):
        clf.predict([[0]])

def test_knn_classifier_feature_mismatch():
    """X_test feature size must match training feature size"""
    clf = KNNClassifier()
    clf.fit(np.array([[0, 1]]), np.array([1]))

    with pytest.raises(ValueError):
        clf.predict([[0]])

def test_knn_classifier_invalid_distance_param():
    """Invalid distance metric raises"""
    with pytest.raises(ValueError):
        KNNClassifier(distance="bad_metric") # type: ignore

def test_knn_classifier_neighbors_exceed_training_size():
    """k > n_samples raises ValueError inside kneighbors"""
    X = np.array([[0], [1]])
    y = np.array([0, 1])

    clf = KNNClassifier(n_neighbors=5)
    clf.fit(X, y)

    with pytest.raises(ValueError):
        clf.kneighbors([[0]])

#------------------------------
## Edge Cases
#------------------------------

def test_knn_classifier_zero_distance_gives_det_proba():
    """Exact match should give probability 1 for the matched class"""
    X = np.array([[5], [10]])
    y = np.array([1, 2])

    clf = KNNClassifier(n_neighbors=2, weights="distance")
    clf.fit(X, y)

    proba = clf.predict_proba([[5]])
    # class "1" should receive full weight
    assert np.isclose(proba[0, 0], 1.0)

def test_knn_classifier_multiple_classes():
    """Classifier handles more than 2 class labels"""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 1, 2])

    clf = KNNClassifier(n_neighbors=3)
    clf.fit(X, y)

    pred = clf.predict([[1.1]])
    assert pred[0] in [0, 1, 2]

#------------------------------
## KNN Classifier Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_knn_regressor_basic_prediction():
    """Simple KNN regression with k=1"""
    X = np.array([[0], [1]])
    y = np.array([10, 20])

    reg = KNNRegressor(n_neighbors=1)
    reg.fit(X, y)

    pred = reg.predict([[0.2]])
    assert np.isclose(pred[0], 10)

def test_knn_regressor_uniform_average():
    """Uniform weighting should average neighbor targets"""
    X = np.array([[0], [2]])
    y = np.array([10, 30])

    reg = KNNRegressor(n_neighbors=2, weights="uniform")
    reg.fit(X, y)

    pred = reg.predict([[1]])
    assert np.isclose(pred[0], 20)

def test_knn_regressor_distance_weights():
    """Closer point should dominate under distance weighting"""
    X = np.array([[0], [10]])
    y = np.array([0, 100])

    reg = KNNRegressor(n_neighbors=2, weights="distance")
    reg.fit(X, y)

    pred = reg.predict([[1]])
    assert pred[0] < 50

def test_knn_regressor_multiple_queries():
    """Regressor can batch-predict"""
    X = np.array([[0], [1], [2]])
    y = np.array([0, 10, 20])

    reg = KNNRegressor(n_neighbors=2)
    reg.fit(X, y)

    preds = reg.predict([[0], [2]])
    assert preds.shape == (2,)

#------------------------------
## Error Handling Tests
#------------------------------

def test_knn_regressor_predict_before_fit():
    """Predicting before fitting raises RuntimeError"""
    reg = KNNRegressor()
    with pytest.raises(RuntimeError):
        reg.predict([[0]])

def test_knn_regressor_feature_mismatch():
    """Feature mismatch raises ValueError"""
    reg = KNNRegressor()
    reg.fit(np.array([[0, 1]]), np.array([1.0]))

    with pytest.raises(ValueError):
        reg.predict([[0]])

def test_knn_regressor_neighbors_exceed_training_size():
    """k > n_samples should raise ValueError"""
    X = np.array([[0], [1]])
    y = np.array([1.0, 2.0])

    reg = KNNRegressor(n_neighbors=5)
    reg.fit(X, y)

    with pytest.raises(ValueError):
        reg.predict([[0]])


#------------------------------
## Edge Cases
#------------------------------

def test_knn_regressor_exact_duplicate_distance():
    """Exact match → prediction equals that label"""
    X = np.array([[5], [10]])
    y = np.array([50.0, 100.0])

    reg = KNNRegressor(n_neighbors=2, weights="distance")
    reg.fit(X, y)

    pred = reg.predict([[5]])
    assert np.isclose(pred[0], 50.0)

def test_knn_regressor_zero_weight_row_handling():
    """When all weights are zero (rare), fallback to mean of neighbors"""
    X = np.array([[1], [1]])       # identical → zero-distance for both
    y = np.array([10.0, 30.0])

    reg = KNNRegressor(n_neighbors=2, weights="distance")
    reg.fit(X, y)

    pred = reg.predict([[1]])
    assert np.isclose(pred[0], 20.0)

def test_knn_regressor_output_dtype_float():
    """Regressor output must always be float"""
    X = np.array([[0], [1]])
    y = np.array([1, 2])

    reg = KNNRegressor(n_neighbors=1)
    reg.fit(X, y)

    pred = reg.predict([[0]])
    assert isinstance(pred[0], float)