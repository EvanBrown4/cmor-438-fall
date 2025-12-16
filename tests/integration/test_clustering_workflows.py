import numpy as np

from rice_ml.unsupervised_learning.k_means_clustering import KMeans
from rice_ml.unsupervised_learning.dbscan import DBSCAN
from rice_ml.unsupervised_learning.community_detection import CommunityDetector


#------------------------------
## Clustering Integration Tests
#------------------------------

def test_kmeans_clustering_blobs():
    """KMeans should identify blob structure"""
    rng = np.random.default_rng(0)

    X1 = rng.normal(loc=0, scale=0.3, size=(100, 2))
    X2 = rng.normal(loc=3, scale=0.3, size=(100, 2))
    X = np.vstack([X1, X2])

    km = KMeans(n_clusters=2)
    labels = km.fit_predict(X)

    assert len(np.unique(labels)) == 2


def test_dbscan_detects_noise():
    """DBSCAN should label noise points"""
    rng = np.random.default_rng(1)

    cluster = rng.normal(size=(100, 2))
    noise = rng.uniform(low=-6, high=6, size=(20, 2))
    X = np.vstack([cluster, noise])

    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X)

    assert -1 in labels


def test_community_detection_basic():
    """Community detection should partition graph"""
    adjacency = np.array([
        [0,1,1,0,0],
        [1,0,1,0,0],
        [1,1,0,0,0],
        [0,0,0,0,1],
        [0,0,0,1,0],
    ])

    cd = CommunityDetector()
    labels = cd.fit_predict(adjacency)

    assert len(np.unique(labels)) == 2
