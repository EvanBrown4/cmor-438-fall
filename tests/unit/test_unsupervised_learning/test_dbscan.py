import pytest
import numpy as np
from src.rice_ml.unsupervised_learning.dbscan import DBSCAN


#------------------------------
## Standard Tests
#------------------------------

def test_dbscan_basic_fit_and_predict():
    """Test basic clustering functionality"""
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [25, 80]])
    
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(X)
    
    assert hasattr(clustering, 'labels_')
    assert len(clustering.labels_) == len(X)


def test_dbscan_two_clear_clusters():
    """Should identify two clear clusters"""
    # Two separated clusters
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],
                  [10, 10], [10, 11], [11, 10], [11, 11]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    # Should find 2 clusters (excluding noise)
    assert clustering.n_clusters_ == 2
    # No noise in this clean example
    assert np.sum(clustering.labels_ == -1) == 0


def test_dbscan_identifies_noise():
    """Should identify noise points correctly"""
    X = np.array([[1, 2], [2, 2], [2, 3],
                  [8, 7], [8, 8], [25, 80]])
    
    clustering = DBSCAN(eps=3, min_samples=2)
    clustering.fit(X)
    
    # Point at [25, 80] should be noise
    assert clustering.labels_[-1] == -1


def test_dbscan_fit_predict():
    """fit_predict should work as convenience method"""
    X = np.array([[1, 1], [2, 1], [10, 10], [11, 10]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    labels = clustering.fit_predict(X)
    
    assert len(labels) == len(X)
    assert np.array_equal(labels, clustering.labels_)


def test_dbscan_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[1, 1], [2, 1]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    result = clustering.fit(X)
    
    assert result is clustering


def test_dbscan_core_samples():
    """Should correctly identify core samples"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],
                  [10, 10]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    # First 4 points should be core samples
    assert len(clustering.core_samples_) >= 1
    assert all(i in clustering.core_samples_ for i in range(4))


def test_dbscan_single_cluster():
    """Densely packed points should form single cluster"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2], [3, 1]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    assert clustering.n_clusters_ == 1
    assert all(clustering.labels_ >= 0)


def test_dbscan_all_noise():
    """Sparse points should all be labeled as noise"""
    X = np.array([[1, 1], [10, 10], [20, 20], [30, 30]])
    
    clustering = DBSCAN(eps=1, min_samples=3)
    clustering.fit(X)
    
    assert clustering.n_clusters_ == 0
    assert all(clustering.labels_ == -1)


def test_dbscan_varying_eps():
    """Different eps values should produce different results"""
    X = np.array([[1, 1], [1, 2], [2, 1],
                  [10, 10], [10, 11], [11, 10]])
    
    # Small eps - may create more clusters or noise
    clustering_small = DBSCAN(eps=1.5, min_samples=2)
    clustering_small.fit(X)
    
    # Large eps - should merge clusters
    clustering_large = DBSCAN(eps=15, min_samples=2)
    clustering_large.fit(X)
    
    # Large eps should produce fewer or equal clusters
    assert clustering_large.n_clusters_ <= clustering_small.n_clusters_


def test_dbscan_varying_min_samples():
    """Different min_samples should produce different results"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    
    # Low min_samples
    clustering_low = DBSCAN(eps=2, min_samples=2)
    clustering_low.fit(X)
    
    # High min_samples - stricter
    clustering_high = DBSCAN(eps=2, min_samples=4)
    clustering_high.fit(X)
    
    # Higher min_samples may create more noise or different clusters
    assert len(clustering_low.core_samples_) >= len(clustering_high.core_samples_)


def test_dbscan_multiple_features():
    """Should work with more than 2 dimensions"""
    X = np.array([[1, 1, 1], [1, 1, 2], [2, 1, 1],
                  [10, 10, 10], [10, 10, 11], [11, 10, 10]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    assert clustering.n_clusters_ >= 1
    assert len(clustering.labels_) == len(X)


def test_dbscan_border_points():
    """Should correctly handle border points"""
    # Create a cluster with clear core and border points
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1],  # Core region
                  [2, 2]])  # Border point
    
    clustering = DBSCAN(eps=2, min_samples=3)
    clustering.fit(X)
    
    # All points should be in one cluster
    assert clustering.n_clusters_ == 1
    assert all(clustering.labels_ >= 0)


def test_dbscan_arbitrary_shape():
    """Should handle non-spherical clusters"""
    # Create a curved cluster
    X = np.array([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                  [4, 1], [4, 2], [4, 3], [4, 4]])
    
    clustering = DBSCAN(eps=1.5, min_samples=2)
    clustering.fit(X)
    
    # Should form a single connected cluster
    assert clustering.n_clusters_ == 1


def test_dbscan_labels_consecutive():
    """Cluster labels should be consecutive starting from 0"""
    X = np.array([[1, 1], [2, 1], [3, 1],
                  [10, 10], [11, 10], [12, 10],
                  [20, 20], [21, 20], [22, 20]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    unique_labels = np.unique(clustering.labels_)
    cluster_labels = unique_labels[unique_labels >= 0]
    
    if len(cluster_labels) > 0:
        assert np.array_equal(cluster_labels, np.arange(len(cluster_labels)))


#------------------------------
## Error Handling Tests
#------------------------------

def test_dbscan_negative_eps():
    """Should raise ValueError for negative eps"""
    X = np.array([[1, 1], [2, 2]])
    clustering = DBSCAN(eps=-1, min_samples=2)
    
    with pytest.raises(ValueError, match="eps must be positive"):
        clustering.fit(X)


def test_dbscan_zero_eps():
    """Should raise ValueError for zero eps"""
    X = np.array([[1, 1], [2, 2]])
    clustering = DBSCAN(eps=0, min_samples=2)
    
    with pytest.raises(ValueError, match="eps must be positive"):
        clustering.fit(X)


def test_dbscan_invalid_min_samples():
    """Should raise ValueError for min_samples < 1"""
    X = np.array([[1, 1], [2, 2]])
    clustering = DBSCAN(eps=1, min_samples=0)
    
    with pytest.raises(ValueError, match="min_samples must be >= 1"):
        clustering.fit(X)


def test_dbscan_unsupported_metric():
    """Should raise ValueError for unsupported metrics"""
    X = np.array([[1, 1], [2, 2]])
    clustering = DBSCAN(eps=1, min_samples=2, metric="cosine") # type: ignore
    
    with pytest.raises(ValueError, match="metric must be"):
        clustering.fit(X)


def test_dbscan_manhattan_metric():
    """Should work with Manhattan distance metric"""
    X = np.array([[1, 1], [2, 1], [10, 10], [11, 10]])
    
    clustering = DBSCAN(eps=2, min_samples=2, metric="manhattan")
    clustering.fit(X)
    
    assert clustering.n_clusters_ >= 1
    assert len(clustering.labels_) == len(X)


def test_dbscan_euclidean_vs_manhattan():
    """Euclidean and Manhattan should produce potentially different results"""
    X = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
    
    clustering_euc = DBSCAN(eps=1.5, min_samples=2, metric="euclidean")
    clustering_euc.fit(X)
    
    clustering_man = DBSCAN(eps=1.5, min_samples=2, metric="manhattan")
    clustering_man.fit(X)
    
    # Both should complete successfully
    assert len(clustering_euc.labels_) == len(X)
    assert len(clustering_man.labels_) == len(X)
    
    # For point (0,0) and (1,1): euclidean = sqrt(2) ≈ 1.41, manhattan = 2
    # So results may differ depending on eps


#------------------------------
## Edge Cases
#------------------------------

def test_dbscan_single_point():
    """Should handle single point"""
    X = np.array([[1, 1]])
    
    clustering = DBSCAN(eps=1, min_samples=1)
    clustering.fit(X)
    
    # Single point with min_samples=1 should form a cluster
    assert clustering.n_clusters_ == 1
    assert clustering.labels_[0] == 0


def test_dbscan_two_points_close():
    """Should handle two close points"""
    X = np.array([[1, 1], [1, 2]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    # Should form one cluster
    assert clustering.n_clusters_ == 1
    assert clustering.labels_[0] == clustering.labels_[1]


def test_dbscan_two_points_far():
    """Should handle two distant points"""
    X = np.array([[1, 1], [100, 100]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    clustering.fit(X)
    
    # Both should be noise
    assert clustering.n_clusters_ == 0
    assert all(clustering.labels_ == -1)


def test_dbscan_identical_points():
    """Should handle duplicate/identical points"""
    X = np.array([[1, 1], [1, 1], [1, 1]])
    
    clustering = DBSCAN(eps=0.1, min_samples=2)
    clustering.fit(X)
    
    # Should form one cluster
    assert clustering.n_clusters_ == 1
    assert all(clustering.labels_ == 0)


def test_dbscan_min_samples_equals_data_size():
    """Should handle min_samples equal to dataset size"""
    X = np.array([[1, 1], [1, 2], [2, 1]])
    
    clustering = DBSCAN(eps=5, min_samples=3)
    clustering.fit(X)
    
    # All points should be in one cluster if close enough
    assert clustering.n_clusters_ == 1


def test_dbscan_empty_array():
    X = np.array([]).reshape(0, 2)

    clustering = DBSCAN(eps=1, min_samples=2)

    with pytest.raises(ValueError, match="Array cannot be empty."):
        clustering.fit(X)


def test_dbscan_high_dimensional():
    """Should work with high-dimensional data"""
    np.random.seed(42)
    X = np.random.randn(20, 10)
    
    clustering = DBSCAN(eps=5, min_samples=3)
    clustering.fit(X)
    
    # Should complete without error
    assert len(clustering.labels_) == 20


def test_dbscan_preserves_order():
    """Labels should correspond to input order"""
    X = np.array([[1, 1], [2, 1], [3, 1], [10, 10]])
    
    clustering = DBSCAN(eps=2, min_samples=2)
    labels = clustering.fit_predict(X)
    
    # First 3 should be in same cluster
    assert labels[0] == labels[1] == labels[2]
    # Last should be different (noise or different cluster)
    assert labels[3] != labels[0] or labels[3] == -1