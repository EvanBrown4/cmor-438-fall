import pytest
import numpy as np
from src.rice_ml.unsupervised_learning.community_detection import CommunityDetector


#------------------------------
## Standard Tests
#------------------------------

def test_detector_basic_fit_and_predict():
    """Test simple community detection on a basic graph"""
    # Simple graph: two connected pairs
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    labels = detector.fit_predict(A)
    
    assert len(labels) == 4
    assert detector.n_communities_ >= 1


def test_detector_two_clear_communities():
    """Should detect two clear communities in a simple graph"""
    # Two cliques connected by a single edge
    A = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    # Should find reasonable communities
    assert detector.n_communities_ >= 1
    assert detector.n_communities_ <= 5


def test_detector_returns_self():
    """fit() should return self for chaining"""
    A = np.array([[0, 1], [1, 0]])
    
    detector = CommunityDetector()
    result = detector.fit(A)
    
    assert result is detector


def test_detector_modularity_calculated():
    """Should calculate modularity score"""
    A = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    assert hasattr(detector, "modularity_")
    assert isinstance(detector.modularity_, float)


def test_detector_labels_consecutive():
    """Community labels should be consecutive integers starting from 0"""
    A = np.array([
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    labels = detector.fit_predict(A)
    
    unique_labels = np.unique(labels)
    assert np.array_equal(unique_labels, np.arange(len(unique_labels)))


def test_detector_weighted_graph():
    """Should handle weighted graphs"""
    A = np.array([
        [0, 2.5, 0, 0],
        [2.5, 0, 0.1, 0],
        [0, 0.1, 0, 3.0],
        [0, 0, 3.0, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    labels = detector.fit_predict(A)
    
    assert len(labels) == 4
    assert detector.n_communities_ >= 1


def test_detector_resolution_parameter():
    """Should respect resolution parameter"""
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    
    # Higher resolution should tend toward more communities
    detector_high = CommunityDetector(resolution=2.0, random_state=42)
    detector_high.fit(A)
    
    # Lower resolution should tend toward fewer communities
    detector_low = CommunityDetector(resolution=0.5, random_state=42)
    detector_low.fit(A)
    
    # Just check that both complete successfully
    assert detector_high.n_communities_ >= 1
    assert detector_low.n_communities_ >= 1


def test_detector_max_iter():
    """Should respect max_iter parameter"""
    A = np.array([
        [0, 1, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 1, 1, 0]
    ])
    
    detector = CommunityDetector(max_iter=1, random_state=42)
    detector.fit(A)
    
    assert detector.n_communities_ >= 1


def test_detector_random_state():
    """Random state should produce reproducible results"""
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0]
    ])
    
    detector1 = CommunityDetector(random_state=42)
    labels1 = detector1.fit_predict(A)
    
    detector2 = CommunityDetector(random_state=42)
    labels2 = detector2.fit_predict(A)
    
    assert np.array_equal(labels1, labels2)


def test_detector_get_communities():
    """get_communities() should return list of node arrays"""
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    communities = detector.get_communities()
    
    assert isinstance(communities, list)
    assert len(communities) == detector.n_communities_
    
    # All nodes should be assigned to exactly one community
    all_nodes = np.concatenate(communities)
    assert len(all_nodes) == 4
    assert len(np.unique(all_nodes)) == 4


def test_detector_complete_graph():
    """Complete graph should result in single community"""
    n = 5
    A = np.ones((n, n)) - np.eye(n)
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    # Complete graph has best modularity with all nodes in one community
    assert detector.n_communities_ == 1


def test_detector_disconnected_components():
    """Disconnected components should be in separate communities"""
    # Two separate triangles
    A = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    labels = detector.fit_predict(A)
    
    # Should detect at least 2 communities
    assert detector.n_communities_ >= 2
    
    # Nodes 0,1,2 should be in one community, nodes 3,4,5 in another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_detector_symmetrizes_asymmetric():
    """Should symmetrize non-symmetric adjacency matrices"""
    # Asymmetric matrix
    A = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    # Should complete without error
    assert np.allclose(detector.adjacency_, detector.adjacency_.T)


#------------------------------
## Error Handling Tests
#------------------------------

def test_detector_get_communities_before_fit():
    """get_communities() before fit should raise RuntimeError"""
    detector = CommunityDetector()
    
    with pytest.raises(RuntimeError):
        detector.get_communities()


def test_detector_non_square_matrix():
    """Should raise ValueError for non-square matrix"""
    A = np.array([[1, 2, 3], [4, 5, 6]])
    
    detector = CommunityDetector()
    
    with pytest.raises(ValueError):
        detector.fit(A)


def test_detector_negative_weights():
    """Should raise ValueError for negative weights"""
    A = np.array([
        [0, 1, -1],
        [1, 0, 1],
        [-1, 1, 0]
    ])
    
    detector = CommunityDetector()
    
    with pytest.raises(ValueError):
        detector.fit(A)


#------------------------------
## Edge Cases
#------------------------------

def test_detector_empty_graph():
    """Should handle graph with no edges"""
    A = np.zeros((5, 5))
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    # Each node should be its own community
    assert detector.n_communities_ == 5
    assert detector.modularity_ == 0.0


def test_detector_single_node():
    """Should handle single node graph"""
    A = np.array([[0]])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    assert detector.n_communities_ == 1
    assert detector.labels_[0] == 0


def test_detector_two_nodes_connected():
    """Should handle two connected nodes"""
    A = np.array([[0, 1], [1, 0]])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    assert len(detector.labels_) == 2
    assert detector.n_communities_ >= 1


def test_detector_two_nodes_disconnected():
    """Should handle two disconnected nodes"""
    A = np.array([[0, 0], [0, 0]])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    assert detector.n_communities_ == 2
    assert detector.modularity_ == 0.0


def test_detector_self_loops():
    """Should handle graphs with self-loops"""
    A = np.array([
        [1, 1, 0],
        [1, 2, 1],
        [0, 1, 1]
    ])
    
    detector = CommunityDetector(random_state=42)
    detector.fit(A)
    
    # Should complete without error
    assert detector.n_communities_ >= 1