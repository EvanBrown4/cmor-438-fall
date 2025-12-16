import numpy as np
import pytest

from rice_ml.unsupervised_learning.community_detection import (
    CommunityDetector,
)

# ---------------------------------------
# Helpers
# ---------------------------------------

def make_two_community_graph():
    """
    Create a simple graph with two obvious communities.
    
    Community A: nodes 0–4
    Community B: nodes 5–9
    """
    n = 10
    adj = np.zeros((n, n))

    # Dense intra-community edges
    for i in range(5):
        for j in range(5):
            if i != j:
                adj[i, j] = 1

    for i in range(5, 10):
        for j in range(5, 10):
            if i != j:
                adj[i, j] = 1

    # Sparse inter-community edges
    adj[2, 7] = adj[7, 2] = 0.1

    return adj


# ---------------------------------------
# Integration Tests
# ---------------------------------------

def test_community_detection_basic_partition():
    """Communities form a valid partition of nodes"""
    adj = make_two_community_graph()

    model = CommunityDetector()
    labels = model.fit_predict(adj)

    # One label per node
    assert len(labels) == adj.shape[0]

    # All nodes assigned
    assert not np.any(labels == -1)

    # Labels are contiguous integers
    unique = np.unique(labels)
    assert set(unique) == set(range(len(unique)))


def test_community_detection_expected_structure():
    """Obvious graph structure should yield two communities"""
    adj = make_two_community_graph()

    model = CommunityDetector()
    labels = model.fit_predict(adj)

    # Expect exactly two communities
    assert len(np.unique(labels)) == 2

    # Nodes 0–4 should match, 5–9 should match
    assert len(set(labels[:5])) == 1
    assert len(set(labels[5:])) == 1
    assert labels[0] != labels[5]


def test_community_detection_determinism():
    """Fixed random_state yields identical partitions"""
    adj = make_two_community_graph()

    model1 = CommunityDetector(random_state=42)
    model2 = CommunityDetector(random_state=42)

    labels1 = model1.fit_predict(adj)
    labels2 = model2.fit_predict(adj)

    np.testing.assert_array_equal(labels1, labels2)


def test_community_detection_pipeline_with_pca():
    """PCA → graph → community detection integration"""
    from rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis

    rng = np.random.default_rng(0)

    # Create clustered feature data
    X = np.vstack([
        rng.normal(loc=0, scale=0.5, size=(50, 10)),
        rng.normal(loc=5, scale=0.5, size=(50, 10)),
    ])

    pca = PrincipalComponentAnalysis(n_components=2)
    X_red = pca.fit_transform(X)

    # Build similarity graph
    dists = np.linalg.norm(X_red[:, None] - X_red[None, :], axis=2)
    adj = np.exp(-dists)

    model = CommunityDetector(random_state=0)
    labels = model.fit_predict(adj)

    assert len(labels) == X.shape[0]
    assert len(np.unique(labels)) >= 2


def test_community_detection_error_propagation():
    """Invalid graph structure propagates clean errors"""
    model = CommunityDetector()

    with pytest.raises(ValueError):
        model.fit_predict(np.array([[1, 2, 3]]))  # Not square
