import numpy as np

from rice_ml.supervised_learning.decision_trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)
from rice_ml.supervised_learning.ensemble_methods import RandomForestClassifier


#------------------------------
## Tree-Based Integration Tests
#------------------------------

def test_decision_tree_classifier_pipeline():
    """Basic classification with decision tree"""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert np.all(np.isin(preds, [0, 1]))


def test_decision_tree_regressor_pipeline():
    """Regression tree learns nonlinear function"""
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(200, 1))
    y = X[:, 0] ** 2

    reg = DecisionTreeRegressor(max_depth=4, random_state=42)
    reg.fit(X, y)

    preds = reg.predict(X)
    mse = np.mean((preds - y) ** 2)

    assert mse < 0.5


def test_random_forest_classifier_pipeline():
    """Random forest improves stability over single tree"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 5))
    y = (X[:, 0] > 0).astype(int)

    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    forest = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)

    tree.fit(X, y)
    forest.fit(X, y)

    tree_acc = np.mean(tree.predict(X) == y)
    forest_acc = np.mean(forest.predict(X) == y)

    assert forest_acc > 0.65