import pytest
import numpy as np
from src.rice_ml.unsupervised_learning.k_means_clustering import KMeans


#------------------------------
## Standard Tests
#------------------------------

def test_kmeans_basic_fit():
    """Test basic k-means clustering"""
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    assert hasattr(kmeans, 'labels_')
    assert hasattr(kmeans, 'cluster_centers_')
    assert len(kmeans.labels_) == len(X)


def test_kmeans_three_clusters():
    """Should identify three clear clusters"""
    X = np.array([[1, 1], [1, 2], [2, 1],
                  [10, 10], [10, 11], [11, 10],
                  [20, 20], [20, 21], [21, 20]])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Should have 3 clusters
    assert len(np.unique(kmeans.labels_)) == 3
    # Centers should be roughly at (1.33, 1.33), (10.33, 10.33), (20.33, 20.33)
    assert kmeans.cluster_centers_.shape == (3, 2)


def test_kmeans_returns_self():
    """fit() should return self for chaining"""
    X = np.array([[1, 1], [2, 2]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    result = kmeans.fit(X)
    
    assert result is kmeans


def test_kmeans_fit_predict():
    """fit_predict should work as convenience method"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)
    
    assert len(labels) == len(X)
    assert np.array_equal(labels, kmeans.labels_)


def test_kmeans_predict():
    """predict() should assign new points to clusters"""
    X_train = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    X_test = np.array([[1.5, 1.5], [10.5, 10.5]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_train)
    labels = kmeans.predict(X_test)
    
    assert len(labels) == len(X_test)
    # Points should be assigned to one of the clusters
    assert all(label in [0, 1] for label in labels)


def test_kmeans_transform():
    """transform() should return distances to cluster centers"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    distances = kmeans.transform(X)
    
    # Should have shape (n_samples, n_clusters)
    assert distances.shape == (4, 2)
    # All distances should be non-negative
    assert np.all(distances >= 0)


def test_kmeans_inertia():
    """Should calculate inertia correctly"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Inertia should be positive
    assert kmeans.inertia_ >= 0
    # More clusters should generally mean lower inertia
    kmeans_more = KMeans(n_clusters=3, random_state=42)
    kmeans_more.fit(X)
    assert kmeans_more.inertia_ <= kmeans.inertia_


def test_kmeans_n_iter():
    """Should track number of iterations"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, max_iter=100, random_state=42)
    kmeans.fit(X)
    
    assert hasattr(kmeans, 'n_iter_')
    assert kmeans.n_iter_ >= 1
    assert kmeans.n_iter_ <= 100


def test_kmeans_convergence():
    """Should converge with tolerance"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    
    kmeans = KMeans(n_clusters=2, tol=1e-4, max_iter=100, random_state=42)
    kmeans.fit(X)
    
    # Should converge before max_iter on this simple data
    assert kmeans.n_iter_ < 100


def test_kmeans_random_init():
    """Should work with random initialization"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, init='random', random_state=42)
    kmeans.fit(X)
    
    assert len(np.unique(kmeans.labels_)) == 2


def test_kmeans_plus_plus_init():
    """Should work with k-means++ initialization"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
    kmeans.fit(X)
    
    assert len(np.unique(kmeans.labels_)) == 2


def test_kmeans_random_state_reproducibility():
    """Random state should produce reproducible results"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2],
                  [10, 10], [10, 11], [11, 10], [11, 11]])
    
    kmeans1 = KMeans(n_clusters=2, random_state=42)
    labels1 = kmeans1.fit_predict(X)
    
    kmeans2 = KMeans(n_clusters=2, random_state=42)
    labels2 = kmeans2.fit_predict(X)
    
    assert np.array_equal(labels1, labels2)


def test_kmeans_multiple_features():
    """Should work with more than 2 dimensions"""
    X = np.array([[1, 1, 1], [1, 1, 2], [2, 1, 1],
                  [10, 10, 10], [10, 10, 11], [11, 10, 10]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    assert kmeans.cluster_centers_.shape == (2, 3)
    assert len(kmeans.labels_) == len(X)


def test_kmeans_single_cluster():
    """Should handle single cluster case"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(X)
    
    assert all(kmeans.labels_ == 0)
    assert kmeans.cluster_centers_.shape == (1, 2)


def test_kmeans_labels_range():
    """Labels should be in range [0, n_clusters)"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11],
                  [20, 20], [20, 21]])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    assert all(0 <= label < 3 for label in kmeans.labels_)


def test_kmeans_max_iter_limit():
    """Should respect max_iter limit"""
    X = np.random.randn(100, 2)
    
    kmeans = KMeans(n_clusters=5, max_iter=5, random_state=42)
    kmeans.fit(X)
    
    assert kmeans.n_iter_ <= 5


def test_kmeans_empty_cluster_handling():
    """Should handle empty clusters by reinitializing"""
    # Create data that might result in empty clusters
    X = np.array([[1, 1], [1, 1.01], [1, 0.99],
                  [10, 10], [10, 10.01]])
    
    # More clusters than natural groupings
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)
    
    # Should complete without error
    assert kmeans.cluster_centers_.shape == (4, 2)


def test_kmeans_well_separated_clusters():
    """Should perfectly separate well-separated clusters"""
    X = np.array([[0, 0], [0, 1], [1, 0],
                  [100, 100], [100, 101], [101, 100]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # First 3 should be in one cluster, last 3 in another
    assert kmeans.labels_[0] == kmeans.labels_[1] == kmeans.labels_[2]
    assert kmeans.labels_[3] == kmeans.labels_[4] == kmeans.labels_[5]
    assert kmeans.labels_[0] != kmeans.labels_[3]


#------------------------------
## Error Handling Tests
#------------------------------

def test_kmeans_predict_before_fit():
    """predict() before fit should raise RuntimeError"""
    kmeans = KMeans(n_clusters=2)
    
    with pytest.raises(RuntimeError, match="not been fit"):
        kmeans.predict([[1, 1]])


def test_kmeans_transform_before_fit():
    """transform() before fit should raise RuntimeError"""
    kmeans = KMeans(n_clusters=2)
    
    with pytest.raises(RuntimeError, match="not been fit"):
        kmeans.transform([[1, 1]])


def test_kmeans_empty_input():
    """Should raise ValueError for empty input"""
    X = np.array([]).reshape(0, 2)
    kmeans = KMeans(n_clusters=2)
    
    with pytest.raises(ValueError, match="Array cannot be empty"):
        kmeans.fit(X)


def test_kmeans_invalid_n_clusters_zero():
    """Should raise ValueError for n_clusters <= 0"""
    X = np.array([[1, 1], [2, 2]])
    kmeans = KMeans(n_clusters=0)
    
    with pytest.raises(ValueError, match="positive integer"):
        kmeans.fit(X)


def test_kmeans_invalid_n_clusters_negative():
    """Should raise ValueError for negative n_clusters"""
    X = np.array([[1, 1], [2, 2]])
    kmeans = KMeans(n_clusters=-1)
    
    with pytest.raises(ValueError, match="positive integer"):
        kmeans.fit(X)


def test_kmeans_n_clusters_exceeds_samples():
    """Should raise ValueError when n_clusters > n_samples"""
    X = np.array([[1, 1], [2, 2]])
    kmeans = KMeans(n_clusters=3)
    
    with pytest.raises(ValueError, match="exceed the number of samples"):
        kmeans.fit(X)


def test_kmeans_invalid_init():
    """Should raise ValueError for invalid init method"""
    X = np.array([[1, 1], [2, 2]])
    kmeans = KMeans(n_clusters=2, init='invalid') # type: ignore
    
    with pytest.raises(ValueError, match="must be 'k-means\\+\\+' or 'random'"):
        kmeans.fit(X)


def test_kmeans_invalid_max_iter():
    """Should raise ValueError for non-positive max_iter"""
    X = np.array([[1, 1], [2, 2]])
    kmeans = KMeans(n_clusters=2, max_iter=0)
    
    with pytest.raises(ValueError, match="max_iter must be positive"):
        kmeans.fit(X)


def test_kmeans_invalid_tol():
    """Should raise ValueError for negative tol"""
    X = np.array([[1, 1], [2, 2]])
    kmeans = KMeans(n_clusters=2, tol=-0.1)
    
    with pytest.raises(ValueError, match="tol must be non-negative"):
        kmeans.fit(X)


#------------------------------
## Edge Cases
#------------------------------

def test_kmeans_identical_points():
    """Should handle all identical points"""
    X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # All centroids should be at the same location
    assert np.allclose(kmeans.cluster_centers_[0], kmeans.cluster_centers_[1])


def test_kmeans_two_points():
    """Should handle two points with two clusters"""
    X = np.array([[0, 0], [10, 10]])
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X)
    
    # Each point should be its own cluster
    assert kmeans.labels_[0] != kmeans.labels_[1]


def test_kmeans_high_dimensional():
    """Should work with high-dimensional data"""
    np.random.seed(42)
    X = np.random.randn(50, 20)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    
    assert kmeans.cluster_centers_.shape == (5, 20)
    assert len(kmeans.labels_) == 50


def test_kmeans_single_point():
    """Should handle single point"""
    X = np.array([[1, 1]])
    
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(X)
    
    assert kmeans.labels_[0] == 0
    assert np.allclose(kmeans.cluster_centers_[0], [1, 1])


def test_kmeans_collinear_points():
    """Should handle collinear points"""
    X = np.array([[i, i] for i in range(10)])
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Should partition the line into 3 segments
    assert len(np.unique(kmeans.labels_)) == 3


def test_kmeans_zero_tolerance():
    """Should work with zero tolerance"""
    X = np.array([[1, 1], [1, 2], [10, 10], [10, 11]])
    
    kmeans = KMeans(n_clusters=2, tol=0.0, max_iter=10, random_state=42)
    kmeans.fit(X)
    
    # Should run for max_iter iterations with tol=0
    assert kmeans.n_iter_ == 10


def test_kmeans_large_tolerance():
    """Should converge quickly with large tolerance"""
    X = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    
    kmeans = KMeans(n_clusters=2, tol=100.0, random_state=42)
    kmeans.fit(X)
    
    # Should converge in very few iterations
    assert kmeans.n_iter_ <= 3