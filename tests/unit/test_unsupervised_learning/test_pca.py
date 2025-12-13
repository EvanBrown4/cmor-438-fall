import pytest
import numpy as np
from src.rice_ml.unsupervised_learning.pca import PrincipalComponentAnalysis


#------------------------------
## Standard Tests
#------------------------------

def test_pca_basic_fit():
    """Test basic PCA fitting"""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    pca = PrincipalComponentAnalysis(n_components=2)
    pca.fit(X)
    
    assert hasattr(pca, 'components_')
    assert hasattr(pca, 'explained_variance_')
    assert pca.components_.shape == (2, 3)


def test_pca_transform():
    """Test PCA transformation"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    
    pca = PrincipalComponentAnalysis(n_components=1)
    pca.fit(X)
    Z = pca.transform(X)
    
    assert Z.shape == (4, 1)


def test_pca_fit_transform():
    """Test fit_transform convenience method"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    pca = PrincipalComponentAnalysis(n_components=1)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (3, 1)
    assert hasattr(pca, 'components_')


def test_pca_inverse_transform():
    """Test inverse transform reconstructs data"""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    
    pca = PrincipalComponentAnalysis(n_components=2)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    
    assert X_reconstructed.shape == X.shape


def test_pca_explained_variance_ratio_sums_to_one():
    """Test explained variance ratios sum to approximately 1"""
    X = np.random.randn(50, 5)
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X)
    
    assert np.allclose(pca.explained_variance_ratio_.sum(), 1.0)


def test_pca_reduces_dimensionality():
    """Test PCA reduces dimensionality correctly"""
    X = np.random.randn(100, 10)
    
    pca = PrincipalComponentAnalysis(n_components=3)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (100, 3)


def test_pca_all_components():
    """Test PCA with all components (n_components=None)"""
    X = np.random.randn(20, 5)
    
    pca = PrincipalComponentAnalysis()
    Z = pca.fit_transform(X)
    
    assert Z.shape[1] == min(20, 5)


def test_pca_whitening():
    """Test PCA with whitening"""
    X = np.random.randn(50, 4)
    
    pca = PrincipalComponentAnalysis(n_components=3, whiten=True)
    Z = pca.fit_transform(X)
    
    # Whitened data should have approximately unit variance
    assert Z.shape == (50, 3)
    variances = np.var(Z, axis=0)
    assert np.allclose(variances, 1.0, atol=0.1)


def test_pca_explained_variance_descending():
    """Test explained variance is in descending order"""
    X = np.random.randn(100, 5)
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X)
    
    # Each component should explain at least as much variance as the next
    for i in range(len(pca.explained_variance_) - 1):
        assert pca.explained_variance_[i] >= pca.explained_variance_[i + 1]


def test_pca_returns_self():
    """Test fit returns self for chaining"""
    X = np.array([[1, 2], [3, 4]])
    
    pca = PrincipalComponentAnalysis()
    result = pca.fit(X)
    
    assert result is pca


def test_pca_preserves_mean():
    """Test that mean is preserved after centering"""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X)
    
    assert np.allclose(pca.mean_, X.mean(axis=0))


def test_pca_orthogonal_components():
    """Test that principal components are orthogonal"""
    X = np.random.randn(50, 5)
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X)
    
    # Components should be orthogonal
    gram = pca.components_ @ pca.components_.T
    assert np.allclose(gram, np.eye(len(pca.components_)), atol=1e-10)


def test_pca_single_component():
    """Test PCA with single component"""
    X = np.random.randn(30, 5)
    
    pca = PrincipalComponentAnalysis(n_components=1)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (30, 1)


def test_pca_deterministic():
    """Test PCA produces deterministic results"""
    X = np.random.randn(20, 4)
    
    pca1 = PrincipalComponentAnalysis(n_components=2)
    Z1 = pca1.fit_transform(X)
    
    pca2 = PrincipalComponentAnalysis(n_components=2)
    Z2 = pca2.fit_transform(X)
    
    # Note: Signs may flip, so check absolute values
    assert np.allclose(np.abs(Z1), np.abs(Z2))


def test_pca_reconstruction_quality():
    """Test reconstruction quality with all components"""
    X = np.random.randn(20, 5)
    
    pca = PrincipalComponentAnalysis()
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    
    # With all components, reconstruction should be nearly perfect
    assert np.allclose(X, X_reconstructed, atol=1e-10)


#------------------------------
## Error Handling Tests
#------------------------------

def test_pca_transform_before_fit():
    """Test transform before fit raises error"""
    pca = PrincipalComponentAnalysis()
    
    with pytest.raises(RuntimeError, match="not been fit"):
        pca.transform([[1, 2, 3]])


def test_pca_inverse_transform_before_fit():
    """Test inverse_transform before fit raises error"""
    pca = PrincipalComponentAnalysis()
    
    with pytest.raises(RuntimeError, match="not been fit"):
        pca.inverse_transform([[1, 2]])


def test_pca_invalid_n_components_zero():
    """Test n_components=0 raises error"""
    X = np.random.randn(10, 5)
    pca = PrincipalComponentAnalysis(n_components=0)
    
    with pytest.raises(ValueError, match="must be between"):
        pca.fit(X)


def test_pca_invalid_n_components_too_large():
    """Test n_components > min(n_samples, n_features) raises error"""
    X = np.random.randn(5, 10)
    pca = PrincipalComponentAnalysis(n_components=20)
    
    with pytest.raises(ValueError, match="must be between"):
        pca.fit(X)


def test_pca_feature_mismatch_transform():
    """Test transform with wrong number of features raises error"""
    X_train = np.random.randn(20, 5)
    X_test = np.random.randn(10, 3)
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X_train)
    
    with pytest.raises(ValueError, match="Expected 5 features"):
        pca.transform(X_test)


def test_pca_component_mismatch_inverse():
    """Test inverse_transform with wrong number of components raises error"""
    X = np.random.randn(20, 5)
    
    pca = PrincipalComponentAnalysis(n_components=3)
    pca.fit(X)
    
    Z_wrong = np.random.randn(10, 2)
    
    with pytest.raises(ValueError, match="incorrect number of components"):
        pca.inverse_transform(Z_wrong)


#------------------------------
## Edge Cases
#------------------------------

def test_pca_single_sample():
    """Test PCA with single sample"""
    X = np.array([[1, 2, 3]])
    
    pca = PrincipalComponentAnalysis(n_components=1)
    pca.fit(X)
    
    # With one sample, explained variance should be 0
    assert pca.explained_variance_[0] == 0.0


def test_pca_two_samples():
    """Test PCA with two samples"""
    X = np.array([[1, 2, 3], [4, 5, 6]])
    
    pca = PrincipalComponentAnalysis(n_components=1)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (2, 1)


def test_pca_constant_feature():
    """Test PCA with constant feature"""
    X = np.array([[1, 5], [2, 5], [3, 5], [4, 5]])
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X)
    
    # Second component should have zero variance
    assert np.allclose(pca.explained_variance_[1], 0.0)


def test_pca_identical_samples():
    """Test PCA with all identical samples"""
    X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    
    pca = PrincipalComponentAnalysis()
    pca.fit(X)
    
    # All variance should be zero
    assert np.allclose(pca.explained_variance_, 0.0)


def test_pca_high_dimensional():
    """Test PCA with high-dimensional data"""
    X = np.random.randn(10, 100)
    
    pca = PrincipalComponentAnalysis(n_components=5)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (10, 5)


def test_pca_more_features_than_samples():
    """Test PCA when n_features > n_samples"""
    X = np.random.randn(5, 20)
    
    pca = PrincipalComponentAnalysis()
    Z = pca.fit_transform(X)
    
    # Can only have min(n_samples, n_features) components
    assert Z.shape[1] == 5


def test_pca_square_matrix():
    """Test PCA with square matrix"""
    X = np.random.randn(10, 10)
    
    pca = PrincipalComponentAnalysis(n_components=5)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (10, 5)


def test_pca_negative_values():
    """Test PCA handles negative values"""
    X = np.random.randn(20, 5) - 10
    
    pca = PrincipalComponentAnalysis(n_components=3)
    Z = pca.fit_transform(X)
    
    assert Z.shape == (20, 3)


def test_pca_whitening_inverse():
    """Test inverse transform works with whitening"""
    X = np.random.randn(30, 5)
    
    pca = PrincipalComponentAnalysis(whiten=True)
    Z = pca.fit_transform(X)
    X_reconstructed = pca.inverse_transform(Z)
    
    # Should reconstruct original data
    assert np.allclose(X, X_reconstructed, atol=1e-10)