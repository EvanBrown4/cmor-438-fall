import pytest
import numpy as np
import pandas as pd
import warnings
from src import normalize

## normalize() testing


#------------------------------
## Basic Functionality Tests
#------------------------------

def test_standard_zscore_2d():
    """Test basic z-score normalization on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="zscore")
    
    assert result.shape == x.shape
    assert isinstance(result, np.ndarray)
    # Check that each column is normalized
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(result.std(axis=0), 1, atol=1e-10)

def test_standard_minmax_2d():
    """Test basic min-max normalization on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="minmax")
    
    assert result.shape == x.shape
    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0

def test_standard_robust_2d():
    """Test basic robust normalization on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape
    assert isinstance(result, np.ndarray)

def test_standard_l1_2d():
    """Test basic L1 normalization on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="l1")
    
    assert result.shape == x.shape
    # Check L1 norm along axis 0
    assert np.allclose(np.sum(np.abs(result), axis=0), 1.0)

def test_standard_l2_2d():
    """Test basic L2 normalization on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="l2")
    
    assert result.shape == x.shape
    # Check L2 norm along axis 0
    assert np.allclose(np.sqrt(np.sum(result**2, axis=0)), 1.0)


#------------------------------
## Dimensionality Tests
#------------------------------

def test_zscore_1d_explicit():
    """Test z-score on 1D array with explicit method"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="zscore")
    
    assert result.shape == x.shape
    assert np.allclose(result.mean(), 0, atol=1e-10)
    assert np.allclose(result.std(), 1, atol=1e-10)

def test_zscore_1d_default():
    """Test z-score on 1D array with default method"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x)  # Default is zscore
    
    assert result.shape == x.shape
    assert np.allclose(result.mean(), 0, atol=1e-10)
    assert np.allclose(result.std(), 1, atol=1e-10)

def test_zscore_2d_explicit():
    """Test z-score on 2D array with explicit method"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x, method="zscore")
    
    assert result.shape == x.shape
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

def test_zscore_2d_default():
    """Test z-score on 2D array with default method"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x)  # Default is zscore
    
    assert result.shape == x.shape
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

def test_minmax_1d():
    """Test min-max on 1D array"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="minmax")
    
    assert result.shape == x.shape
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)

def test_minmax_2d():
    """Test min-max on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x, method="minmax")
    
    assert result.shape == x.shape

def test_robust_1d():
    """Test robust on 1D array"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape

def test_robust_2d():
    """Test robust on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape

def test_l1_1d():
    """Test L1 on 1D array"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = normalize(x, method="l1")
    
    assert result.shape == x.shape
    assert np.isclose(np.sum(np.abs(result)), 1.0)

def test_l1_2d():
    """Test L1 on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="l1")
    
    assert result.shape == x.shape

def test_l2_1d():
    """Test L2 on 1D array"""
    x = np.array([3.0, 4.0])
    result = normalize(x, method="l2")
    
    assert result.shape == x.shape
    assert np.isclose(np.sqrt(np.sum(result**2)), 1.0)

def test_l2_2d():
    """Test L2 on 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="l2")
    
    assert result.shape == x.shape

def test_1d_array():
    """Test with 1D array"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="zscore")
    
    assert result.ndim == 1
    assert result.shape == x.shape

def test_2d_array():
    """Test with 2D array"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x, method="zscore")
    
    assert result.ndim == 2
    assert result.shape == x.shape

def test_0d_array():
    """Test with 0D array (scalar)"""
    x = np.array(5.0)
    
    with pytest.raises(ValueError, match="normalize currently supports only 1D or 2D"):
        normalize(x)

def test_3d_array():
    """Test with 3D array"""
    x = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    
    with pytest.raises(ValueError, match="normalize currently supports only 1D or 2D"):
        normalize(x)

def test_axis_1_with_1d_array():
    """Test that axis=1 with 1D array raises error"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="Axis cannot be 1 for a 1-dimensional array"):
        normalize(x, axis=1)


#------------------------------
## feature_range Parameter Tests
#------------------------------

def test_minmax_feature_range_none():
    """Test minmax with feature_range=None (should default to (0, 1))"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="minmax", feature_range=None)
    
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)

def test_minmax_feature_range_custom():
    """Test minmax with custom feature_range"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="minmax", feature_range=(-1.0, 1.0))
    
    assert np.isclose(result.min(), -1.0)
    assert np.isclose(result.max(), 1.0)

def test_minmax_feature_range_negative():
    """Test minmax with negative range"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="minmax", feature_range=(-5.0, -1.0))
    
    assert np.isclose(result.min(), -5.0)
    assert np.isclose(result.max(), -1.0)

def test_minmax_feature_range_large():
    """Test minmax with large range"""
    x = np.array([1.0, 2.0, 3.0])
    result = normalize(x, method="minmax", feature_range=(0, 100))
    
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 100.0)

def test_feature_range_given_for_non_minmax():
    """Test that feature_range with non-minmax method raises warning"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = normalize(x, method="zscore", feature_range=(0, 1))
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "feature_range" in str(w[-1].message).lower()


#------------------------------
## axis Parameter Tests
#------------------------------

def test_axis_none_1d():
    """Test axis=None on 1D array"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="zscore", axis=None)
    
    assert result.shape == x.shape
    assert np.allclose(result.mean(), 0, atol=1e-10)

def test_axis_none_2d():
    """Test axis=None on 2D array (should default to 0)"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="zscore", axis=None)
    
    assert result.shape == x.shape
    # Should normalize per column (axis=0)
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

def test_axis_0_2d():
    """Test axis=0 on 2D array (normalize per column)"""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    result = normalize(x, method="zscore", axis=0)
    
    assert result.shape == x.shape
    # Each column should have mean 0
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)

def test_axis_1_2d():
    """Test axis=1 on 2D array (normalize per row)"""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = normalize(x, method="zscore", axis=1)
    
    assert result.shape == x.shape
    # Each row should have mean 0
    assert np.allclose(result.mean(axis=1), 0, atol=1e-10)


#------------------------------
## Invalid Inputs Tests
#------------------------------

def test_invalid_x_empty_array():
    """Test with empty array"""
    x = np.array([])
    
    with pytest.raises(ValueError, match="Cannot normalize a 0-length"):
        normalize(x)

def test_invalid_x_empty_matrix():
    """Test with empty matrix"""
    x = np.array([[]]).reshape(0, 2)
    
    with pytest.raises(ValueError, match="Cannot normalize a 0-length"):
        normalize(x)

def test_invalid_method():
    """Test with invalid method"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="'method' must be one of"):
        normalize(x, method="invalid") # type: ignore

def test_invalid_axis_value():
    """Test with invalid axis value"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="'axis' must be 0, 1, or None"):
        normalize(x, axis=2) # type: ignore

def test_invalid_axis_negative():
    """Test with negative axis value"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="'axis' must be 0, 1, or None"):
        normalize(x, axis=-1) # type: ignore

def test_invalid_feature_range_not_tuple():
    """Test with feature_range not a tuple"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(TypeError, match="'feature_range' must be a tuple"):
        normalize(x, method="minmax", feature_range=[0, 1]) # type: ignore

def test_invalid_feature_range_wrong_length():
    """Test with feature_range of wrong length"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(TypeError, match="'feature_range' must be a tuple of two numbers"):
        normalize(x, method="minmax", feature_range=(0, 1, 2)) # type: ignore

def test_invalid_feature_range_non_numeric():
    """Test with non-numeric feature_range elements"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(TypeError, match="'feature_range' values must be numeric"):
        normalize(x, method="minmax", feature_range=("0", "1")) # type: ignore

def test_invalid_feature_range_min_greater_than_max():
    """Test with min > max in feature_range"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="Min value for feature_range must be less than max"):
        normalize(x, method="minmax", feature_range=(1.0, 0.0))

def test_invalid_feature_range_min_equals_max():
    """Test with min = max in feature_range"""
    x = np.array([1.0, 2.0, 3.0])
    
    with pytest.raises(ValueError, match="Min value for feature_range must be less than max"):
        normalize(x, method="minmax", feature_range=(1.0, 1.0))


#------------------------------
## Zero Division Tests
#------------------------------

def test_zscore_zero_stddev_1d():
    """Test z-score when std dev is 0 (constant array) 1D"""
    x = np.array([5.0, 5.0, 5.0, 5.0])
    result = normalize(x, method="zscore")
    
    # When std=0, result should be all zeros (x - mean = 0)
    # But dividing by 0 std gives nan or inf
    assert result.shape == x.shape
    # Check behavior - typically would be nan
    assert np.all(np.isnan(result)) or np.all(result == 0)

def test_zscore_zero_stddev_2d_axis0():
    """Test z-score when std dev is 0 for some columns (2D, axis=0)"""
    x = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    result = normalize(x, method="zscore", axis=0)
    
    assert result.shape == x.shape
    # First column has std=0, second doesn't
    assert np.all(np.isnan(result[:, 0])) or np.all(result[:, 0] == 0)
    assert np.allclose(result[:, 1].mean(), 0, atol=1e-10)

def test_zscore_zero_stddev_2d_axis1():
    """Test z-score when std dev is 0 for some rows (2D, axis=1)"""
    x = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
    result = normalize(x, method="zscore", axis=1)
    
    assert result.shape == x.shape
    # First row has std=0
    assert np.all(np.isnan(result[0, :])) or np.all(result[0, :] == 0)

def test_minmax_zero_range_1d():
    """Test minmax when range is 0 (constant array) 1D"""
    x = np.array([5.0, 5.0, 5.0, 5.0])
    result = normalize(x, method="minmax")
    
    assert result.shape == x.shape
    # When range=0, implementation prevents division by zero
    # Should return lower bound of feature_range
    assert np.allclose(result, 0.0)

def test_minmax_zero_range_2d_axis0():
    """Test minmax when range is 0 for some columns (2D, axis=0)"""
    x = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
    result = normalize(x, method="minmax", axis=0)
    
    assert result.shape == x.shape
    # First column has range=0
    assert np.allclose(result[:, 0], 0.0)
    # Second column should be normalized
    assert np.isclose(result[:, 1].min(), 0.0)
    assert np.isclose(result[:, 1].max(), 1.0)

def test_minmax_zero_range_2d_axis1():
    """Test minmax when range is 0 for some rows (2D, axis=1)"""
    x = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
    result = normalize(x, method="minmax", axis=1)
    
    assert result.shape == x.shape
    # First row has range=0
    assert np.allclose(result[0, :], 0.0)

def test_robust_zero_iqr_1d():
    """Test robust when IQR is 0 (1D)"""
    x = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape
    # When IQR=0, implementation prevents division by zero
    assert np.allclose(result, 0.0)

def test_robust_zero_iqr_2d_axis0():
    """Test robust when IQR is 0 for some columns (2D, axis=0)"""
    x = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0], [5.0, 4.0]])
    result = normalize(x, method="robust", axis=0)
    
    assert result.shape == x.shape
    # First column has IQR=0
    assert np.allclose(result[:, 0], 0.0)

def test_robust_zero_iqr_2d_axis1():
    """Test robust when IQR is 0 for some rows (2D, axis=1)"""
    x = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
    result = normalize(x, method="robust", axis=1)
    
    assert result.shape == x.shape
    # First row has IQR=0
    assert np.allclose(result[0, :], 0.0)

def test_l1_zero_sum_1d():
    """Test L1 when sum of absolute values is 0 (all zeros) 1D"""
    x = np.array([0.0, 0.0, 0.0, 0.0])
    result = normalize(x, method="l1")
    
    assert result.shape == x.shape
    # When sum=0, implementation prevents division by zero
    assert np.allclose(result, 0.0)

def test_l1_zero_sum_2d_axis0():
    """Test L1 when sum is 0 for some columns (2D, axis=0)"""
    x = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    result = normalize(x, method="l1", axis=0)
    
    assert result.shape == x.shape
    # First column has sum=0
    assert np.allclose(result[:, 0], 0.0)
    # Second column should be normalized
    assert np.isclose(np.sum(np.abs(result[:, 1])), 1.0)

def test_l1_zero_sum_2d_axis1():
    """Test L1 when sum is 0 for some rows (2D, axis=1)"""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    result = normalize(x, method="l1", axis=1)
    
    assert result.shape == x.shape
    # First row has sum=0
    assert np.allclose(result[0, :], 0.0)

def test_l2_zero_sum_squares_1d():
    """Test L2 when sum of squares is 0 (all zeros) 1D"""
    x = np.array([0.0, 0.0, 0.0, 0.0])
    result = normalize(x, method="l2")
    
    assert result.shape == x.shape
    # When sum=0, implementation prevents division by zero
    assert np.allclose(result, 0.0)

def test_l2_zero_sum_squares_2d_axis0():
    """Test L2 when sum of squares is 0 for some columns (2D, axis=0)"""
    x = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    result = normalize(x, method="l2", axis=0)
    
    assert result.shape == x.shape
    # First column has sum of squares=0
    assert np.allclose(result[:, 0], 0.0)
    # Second column should be normalized
    assert np.isclose(np.sqrt(np.sum(result[:, 1]**2)), 1.0)

def test_l2_zero_sum_squares_2d_axis1():
    """Test L2 when sum of squares is 0 for some rows (2D, axis=1)"""
    x = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    result = normalize(x, method="l2", axis=1)
    
    assert result.shape == x.shape
    # First row has sum of squares=0
    assert np.allclose(result[0, :], 0.0)


#------------------------------
## Different x Data Type Tests
#------------------------------

def test_non_numeric_dtype_string():
    """Test with string array"""
    x = np.array(["1", "2", "3"])
    
    with pytest.raises(ValueError, match="Cannot normalize a non-number array"):
        normalize(x)

def test_non_numeric_dtype_object():
    """Test with object array"""
    x = np.array([{"a": 1}, {"a": 2}])
    
    with pytest.raises(ValueError, match="Cannot normalize a non-number array"):
        normalize(x)

def test_integer_input_returns_float():
    """Test that integer input returns float output"""
    x = np.array([1, 2, 3, 4, 5])
    result = normalize(x, method="zscore")
    
    assert result.dtype == np.float64

def test_integer_2d_input_returns_float():
    """Test that integer 2D input returns float output"""
    x = np.array([[1, 2], [3, 4], [5, 6]])
    result = normalize(x, method="minmax")
    
    assert result.dtype == np.float64

def test_mixed_int_float_list():
    """Test with mixed int/float list"""
    x = [1, 2.0, 3, 4.0, 5]
    result = normalize(x, method="zscore")
    
    assert result.dtype == np.float64

def test_nan_in_array():
    """Test with NaN values"""
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    result = normalize(x, method="zscore")
    
    # NaN should propagate
    assert np.isnan(result).any()

def test_inf_in_array():
    """Test with inf values"""
    x = np.array([1.0, 2.0, np.inf, 4.0, 5.0])
    result = normalize(x, method="zscore")
    
    # Inf should affect the result
    assert np.isinf(result).any() or np.isnan(result).any()

def test_negative_inf_in_array():
    """Test with negative inf values"""
    x = np.array([1.0, 2.0, -np.inf, 4.0, 5.0])
    result = normalize(x, method="minmax")
    
    assert np.isinf(result).any() or np.isnan(result).any()


#------------------------------
## Non-square Array Behavior Tests
#------------------------------

def test_non_square_axis0():
    """Test broadcasting with non-square array, axis=0"""
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0]])  # 3x5
    result = normalize(x, method="zscore", axis=0)
    
    assert result.shape == (3, 5)
    # Each column should be normalized independently
    for col in range(5):
        assert np.allclose(result[:, col].mean(), 0, atol=1e-10)
        assert np.allclose(result[:, col].std(), 1, atol=1e-10)

def test_non_square_axis1():
    """Test broadcasting with non-square array, axis=1"""
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0]])  # 3x5
    result = normalize(x, method="zscore", axis=1)
    
    assert result.shape == (3, 5)
    # Each row should be normalized independently
    for row in range(3):
        assert np.allclose(result[row, :].mean(), 0, atol=1e-10)
        assert np.allclose(result[row, :].std(), 1, atol=1e-10)

def test_non_square_minmax_axis0():
    """Test minmax with non-square array, axis=0"""
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0]])  # 3x5
    result = normalize(x, method="minmax", axis=0)
    
    assert result.shape == (3, 5)
    # Each column should be normalized independently
    for col in range(5):
        assert np.isclose(result[:, col].min(), 0.0)
        assert np.isclose(result[:, col].max(), 1.0)

def test_non_square_minmax_axis1():
    """Test minmax with non-square array, axis=1"""
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                    [2.0, 3.0, 4.0, 5.0, 6.0],
                    [3.0, 4.0, 5.0, 6.0, 7.0]])  # 3x5
    result = normalize(x, method="minmax", axis=1)
    
    assert result.shape == (3, 5)
    # Each row should be normalized independently
    for row in range(3):
        assert np.isclose(result[row, :].min(), 0.0)
        assert np.isclose(result[row, :].max(), 1.0)


#------------------------------
## Constant Column/Row Tests
#------------------------------

def test_minmax_constant_column_axis0():
    """Test minmax where one column is constant, others vary (axis=0)"""
    x = np.array([[7.0, 1.0, 5.0],
                    [7.0, 2.0, 10.0],
                    [7.0, 3.0, 15.0],
                    [7.0, 4.0, 20.0]])  # First column all 7s
    result = normalize(x, method="minmax", axis=0, feature_range=(0, 1))
    
    assert result.shape == x.shape
    # First column is constant, should map to lower bound (0)
    assert np.allclose(result[:, 0], 0.0)
    # Second column varies, should be normalized
    assert np.isclose(result[:, 1].min(), 0.0)
    assert np.isclose(result[:, 1].max(), 1.0)
    # Third column varies, should be normalized
    assert np.isclose(result[:, 2].min(), 0.0)
    assert np.isclose(result[:, 2].max(), 1.0)

def test_minmax_constant_row_axis1():
    """Test minmax where one row is constant, others vary (axis=1)"""
    x = np.array([[5.0, 5.0, 5.0, 5.0],
                    [1.0, 2.0, 3.0, 4.0],
                    [10.0, 20.0, 30.0, 40.0]])  # First row all 5s
    result = normalize(x, method="minmax", axis=1, feature_range=(0, 1))
    
    assert result.shape == x.shape
    # First row is constant, should map to lower bound (0)
    assert np.allclose(result[0, :], 0.0)
    # Second row varies, should be normalized
    assert np.isclose(result[1, :].min(), 0.0)
    assert np.isclose(result[1, :].max(), 1.0)
    # Third row varies, should be normalized
    assert np.isclose(result[2, :].min(), 0.0)
    assert np.isclose(result[2, :].max(), 1.0)

def test_zscore_constant_column_axis0():
    """Test zscore where one column is constant (axis=0)"""
    x = np.array([[7.0, 1.0],
                    [7.0, 2.0],
                    [7.0, 3.0],
                    [7.0, 4.0]])
    result = normalize(x, method="zscore", axis=0)
    
    assert result.shape == x.shape
    # First column has std=0, should be nan or 0
    assert np.all(np.isnan(result[:, 0])) or np.all(result[:, 0] == 0)
    # Second column should be normalized
    assert np.allclose(result[:, 1].mean(), 0, atol=1e-10)


#------------------------------
## Robust Normalization With Very Small Sample Tests
#------------------------------

def test_robust_size_1():
    """Test robust normalization with single element"""
    x = np.array([5.0])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape
    # With one element, median=element, IQR=0
    assert np.allclose(result, 0.0)

def test_robust_size_2_1d():
    """Test robust normalization with 2 elements (1D)"""
    x = np.array([1.0, 5.0])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape
    # Should handle gracefully
    assert not np.any(np.isinf(result))

def test_robust_size_2_2d_axis0():
    """Test robust normalization with 2 rows (2D, axis=0)"""
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = normalize(x, method="robust", axis=0)
    
    assert result.shape == x.shape
    # Should handle gracefully
    assert not np.any(np.isinf(result))

def test_robust_size_2_2d_axis1():
    """Test robust normalization with 2 columns (2D, axis=1)"""
    x = np.array([[1.0, 3.0]])
    result = normalize(x, method="robust", axis=1)
    
    assert result.shape == x.shape
    # Should handle gracefully
    assert not np.any(np.isinf(result))

def test_robust_size_3():
    """Test robust normalization with 3 elements"""
    x = np.array([1.0, 2.0, 3.0])
    result = normalize(x, method="robust")
    
    assert result.shape == x.shape
    # Median is 2.0, Q1=1.0, Q3=3.0, IQR=2.0
    # (x - 2) / 2
    expected = np.array([-0.5, 0.0, 0.5])
    assert np.allclose(result, expected)


#------------------------------
## Additional Edge Case Tests
#------------------------------

def test_single_row_2d():
    """Test with single row 2D array"""
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
    result = normalize(x, method="zscore", axis=0)
    
    assert result.shape == x.shape
    # With one row, std per column is 0
    assert np.all(np.isnan(result)) or np.all(result == 0)

def test_single_column_2d():
    """Test with single column 2D array"""
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    result = normalize(x, method="zscore", axis=0)
    
    assert result.shape == x.shape
    assert np.allclose(result.mean(), 0, atol=1e-10)
    assert np.allclose(result.std(), 1, atol=1e-10)

def test_minmax_with_negative_values():
    """Test minmax with negative values"""
    x = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
    result = normalize(x, method="minmax")
    
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)

def test_minmax_custom_range_with_floats():
    """Test minmax with float feature_range"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="minmax", feature_range=(0.5, 2.5))
    
    assert np.isclose(result.min(), 0.5)
    assert np.isclose(result.max(), 2.5)

def test_minmax_custom_range_with_ints():
    """Test minmax with integer feature_range"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize(x, method="minmax", feature_range=(0, 10))
    
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 10.0)

def test_list_input():
    """Test with list input"""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = normalize(x, method="zscore")
    
    assert isinstance(result, np.ndarray)
    assert np.allclose(result.mean(), 0, atol=1e-10)

def test_tuple_input():
    """Test with tuple input"""
    x = (1.0, 2.0, 3.0, 4.0, 5.0)
    result = normalize(x, method="zscore")
    
    assert isinstance(result, np.ndarray)
    assert np.allclose(result.mean(), 0, atol=1e-10)

def test_nested_list_input():
    """Test with nested list input"""
    x = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    result = normalize(x, method="minmax")
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 2)

def test_very_large_values():
    """Test with very large values"""
    x = np.array([1e10, 2e10, 3e10, 4e10, 5e10])
    result = normalize(x, method="zscore")
    
    assert np.allclose(result.mean(), 0, atol=1e-5)
    assert np.allclose(result.std(), 1, atol=1e-5)

def test_very_small_values():
    """Test with very small values"""
    x = np.array([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
    result = normalize(x, method="minmax")
    
    assert np.isclose(result.min(), 0.0)
    assert np.isclose(result.max(), 1.0)

def test_all_methods_preserve_shape():
    """Test that all methods preserve input shape"""
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    
    for method in ["zscore", "minmax", "robust", "l1", "l2"]:
        result = normalize(x, method=method)
        assert result.shape == x.shape