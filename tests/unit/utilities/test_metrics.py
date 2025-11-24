import pytest
import numpy as np
import math
import re
from src.utilities.metrics import euclidean_dist, manhattan_dist


#------------------------------
## Standard Tests
#------------------------------

def test_euclidean_dist_1d_basic():
    """Test basic 1D euclidean distance"""
    x = np.array([0, 0])
    y = np.array([3, 4])
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 5.0)

def test_euclidean_dist_2d_basic():
    """Test basic 2D euclidean distance with default axis"""
    x = np.array([[0, 0], [0, 1]])
    y = np.array([[1, 1], [1, 2]])
    result = euclidean_dist(x, y)
    
    assert result.shape == (2,) # type: ignore
    assert np.allclose(result, [np.sqrt(2), np.sqrt(2)])

def test_euclidean_dist_2d_axis_0():
    """Test 2D euclidean distance along axis 0"""
    x = np.array([[0, 0], [0, 1]])
    y = np.array([[1, 1], [1, 2]])
    result = euclidean_dist(x, y, axis=0)
    
    assert result.shape == (2,) # type: ignore
    assert np.allclose(result, [np.sqrt(2), np.sqrt(2)])

def test_euclidean_dist_2d_axis_1():
    """Test 2D euclidean distance along axis 1"""
    x = np.array([[0, 0], [0, 1]])
    y = np.array([[1, 1], [1, 2]])
    result = euclidean_dist(x, y, axis=1)
    
    assert result.shape == (2,)  # type: ignore
    assert np.allclose(result, [np.sqrt(2), np.sqrt(2)])

#------------------------------
## Zero Value Tests
#------------------------------

def test_euclidean_dist_identical_arrays_1d():
    """Test distance between identical 1D arrays (should be 0)"""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    result = euclidean_dist(x, y)
    
    assert np.isclose(result, 0.0)

def test_euclidean_dist_identical_arrays_2d():
    """Test distance between identical 2D arrays (should be 0)"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    result = euclidean_dist(x, y)
    
    assert np.allclose(result, [0.0, 0.0])

def test_euclidean_dist_all_zeros_1d():
    """Test distance between zero arrays"""
    x = np.array([0, 0, 0])
    y = np.array([0, 0, 0])
    result = euclidean_dist(x, y)
    
    assert np.isclose(result, 0.0)

def test_euclidean_dist_all_zeros_2d():
    """Test distance between zero 2D arrays"""
    x = np.array([[0, 0], [0, 0]])
    y = np.array([[0, 0], [0, 0]])
    result = euclidean_dist(x, y)
    
    assert np.allclose(result, [0.0, 0.0])

#------------------------------
## Constraint Tests: Scalar Inputs
#------------------------------

def test_euclidean_dist_scalar_fails():
    """Test that scalar raises ValueError"""
    x = np.array(5)
    y = np.array([1, 2])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 0 dimensions.\n2D validation error: Expected 2D. Got 0 dimensions."):
        euclidean_dist(y, x)

def test_euclidean_dist_both_scalars_fail():
    """Test that both scalars raise ValueError"""
    x = np.array(5)
    y = np.array(3)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 0 dimensions.\n2D validation error: Expected 2D. Got 0 dimensions."):
        euclidean_dist(x, y)

#------------------------------
## Constraint Tests: Dimensionality
#------------------------------

def test_euclidean_dist_1d_passes():
    """Test that 1D arrays pass dimensionality check"""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_euclidean_dist_2d_passes():
    """Test that 2D arrays pass dimensionality check"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    result = euclidean_dist(x, y)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_euclidean_dist_3d_fails():
    """Test that 3D arrays raise ValueError"""
    x = np.array([[[1, 2], [3, 4]]])
    y = np.array([[[5, 6], [7, 8]]])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 3 dimensions.\n2D validation error: Expected 2D. Got 3 dimensions."):
        euclidean_dist(x, y)

def test_euclidean_dist_4d_fails():
    """Test that 4D arrays raise ValueError"""
    x = np.ones((2, 2, 2, 2))
    y = np.ones((2, 2, 2, 2))
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 4 dimensions.\n2D validation error: Expected 2D. Got 4 dimensions."):
        euclidean_dist(x, y)

#------------------------------
## Constraint Tests: Empty Arrays
#------------------------------

def test_euclidean_dist_empty_fails():
    """Test that empty raises ValueError"""
    x = np.array([])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array cannot be empty.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)

    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array cannot be empty.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(y, x)

def test_euclidean_dist_both_empty_fail():
    """Test that both empty arrays raise ValueError"""
    x = np.array([])
    y = np.array([])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array cannot be empty.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)

def test_euclidean_dist_empty_2d_fails():
    """Test that empty 2D raises ValueError"""
    x = np.array([[]]).reshape(0, 2)
    y = np.array([[1, 2]])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 2 dimensions.\n2D validation error: Array cannot be empty."):
        euclidean_dist(x, y)

    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 2 dimensions.\n2D validation error: Array cannot be empty."):
        euclidean_dist(y, x)

#------------------------------
## Constraint Tests: Shape Mismatch
#------------------------------

def test_euclidean_dist_different_lengths_1d_fails():
    """Test that 1D arrays with different lengths raise ValueError"""
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    
    with pytest.raises(ValueError, match=re.escape("x and y must have the same shape, got (3,) and (2,).")):
        euclidean_dist(x, y)

def test_euclidean_dist_different_shapes_2d_fails():
    """Test that 2D arrays with different shapes raise ValueError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    
    with pytest.raises(ValueError, match=re.escape("x and y must have the same shape, got (2, 2) and (2, 3).")):
        euclidean_dist(x, y)

def test_euclidean_dist_same_size_different_shape_fails():
    """Test that arrays with same size but different shape raise ValueError"""
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match=re.escape("x and y must have the same shape, got (1, 4) and (2, 2).")):
        euclidean_dist(x, y)

def test_euclidean_dist_matching_shapes_passes():
    """Test that matching shapes pass constraint"""
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[7, 8], [9, 10], [11, 12]])
    result = euclidean_dist(x, y)
    
    assert result.shape == (2,)  # type: ignore

#------------------------------
## Constraint Tests: Non-Numeric Values
#------------------------------

def test_euclidean_dist_string_fails():
    """Test that string values raise TypeError"""
    x = np.array(['a', 'b', 'c'])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must contain numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must contain numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(y, x)

def test_euclidean_dist_object_array_mixed_fails():
    """Test that object array with non-numeric raises TypeError"""
    x = np.array([1, 2, 'c'], dtype=object)
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array contains non-numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)

    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array contains non-numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(y, x)

def test_euclidean_dist_numeric_object_array_passes():
    """Test that object array with only numeric values passes"""
    x = np.array([1, 2, 3], dtype=object)
    y = np.array([4, 5, 6], dtype=object)
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, np.sqrt(27))

#------------------------------
## Constraint Tests: NaN and Infinite Values
#------------------------------

def test_euclidean_dist_nan_fails():
    """Test that NaN raises ValueError"""
    x = np.array([1, np.nan, 3])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(y, x)

def test_euclidean_dist_inf_fails():
    """Test that infinity raises a ValueError"""
    x = np.array([1, np.inf, 3])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(y, x)

def test_euclidean_dist_neg_inf_in_x_fails():
    """Test that negative infinity in x raises ValueError"""
    x = np.array([1, -np.inf, 3])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        euclidean_dist(x, y)

def test_euclidean_dist_finite_values_pass():
    """Test that finite numeric values pass constraint"""
    x = np.array([1.5, -2.3, 0.0, 1e10, -1e10])
    y = np.array([2.5, -3.3, 1.0, 2e10, -2e10])
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isfinite(result)

#------------------------------
## Constraint Tests: Axis Parameter
#------------------------------

def test_euclidean_dist_axis_none():
    """Test axis=None arrays"""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    result = euclidean_dist(x, y, axis=None)
    
    assert isinstance(result, float)
    assert np.isclose(result, np.sqrt(27))


    x2 = np.array([[0, 0],
                   [0, 1]])
    y2 = np.array([[1, 1],
                   [1, 2]])
    result = euclidean_dist(x2, y2, axis=None)    

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_euclidean_dist_axis_negative_valid():
    """Test negative axis indexing (valid)"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    result = euclidean_dist(x, y, axis=-1)
    
    assert result.shape == (2,) # type: ignore
    assert np.allclose(result, [np.sqrt(32), np.sqrt(32)])

    result = euclidean_dist(x, y, axis=-2)
    assert result.shape == (2,) # type: ignore

def test_euclidean_dist_axis_out_of_bounds_fails():
    """Test axis out of bounds (too large) raises ValueError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    
    with pytest.raises(ValueError, match="'axis'=2 out of bounds"):
        euclidean_dist(x, y, axis=2)
    
    with pytest.raises(ValueError, match="'axis'=-3 out of bounds"):
        euclidean_dist(x, y, axis=-3)

def test_euclidean_dist_axis_non_integer_fails():
    """Test non-integer axis raises TypeError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    
    with pytest.raises(TypeError, match="'axis' must be an integer or None"):
        euclidean_dist(x, y, axis=1.5) # type: ignore

def test_euclidean_dist_axis_string_fails():
    """Test string axis raises TypeError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    
    with pytest.raises(TypeError, match="'axis' must be an integer or None"):
        euclidean_dist(x, y, axis='0') # type: ignore

#------------------------------
## Edge Case Tests
#------------------------------

def test_euclidean_dist_single_element_1d():
    """Test 1D arrays with single element"""
    x = np.array([5.0])
    y = np.array([2.0])
    result = euclidean_dist(x, y)
    
    assert np.isclose(result, 3.0)

def test_euclidean_dist_single_element_2d():
    """Test 2D arrays with single element"""
    x = np.array([[5.0]])
    y = np.array([[2.0]])
    result = euclidean_dist(x, y)
    
    assert result.shape == (1,) # type: ignore
    assert np.isclose(result[0], 3.0) # type: ignore

def test_euclidean_dist_large_values():
    """Test with very large values"""
    x = np.array([1e100, 1e100])
    y = np.array([2e100, 2e100])
    result = euclidean_dist(x, y)
    
    assert np.isfinite(result)
    assert result > 0

def test_euclidean_dist_very_small_values():
    """Test with very small values"""
    x = np.array([1e-100, 1e-100])
    y = np.array([2e-100, 2e-100])
    result = euclidean_dist(x, y)
    
    assert np.isfinite(result)
    assert result >= 0

def test_euclidean_dist_negative_values():
    """Test with negative values"""
    x = np.array([-5, -10, -15])
    y = np.array([-2, -8, -12])
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_euclidean_dist_mixed_signs():
    """Test with mixed positive and negative values"""
    x = np.array([-5, 10, -15])
    y = np.array([2, -8, 12])
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_euclidean_dist_integer_arrays():
    """Test with integer arrays"""
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, np.sqrt(27))

def test_euclidean_dist_float_arrays():
    """Test with float arrays"""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, np.sqrt(27))

def test_euclidean_dist_mixed_dtypes():
    """Test with mixed integer and float arrays"""
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4.5, 5.5, 6.5], dtype=np.float64)
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_euclidean_dist_list_inputs():
    """Test with list inputs (should be converted to arrays)"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, np.sqrt(27))

def test_euclidean_dist_tuple_inputs():
    """Test with tuple inputs (should be converted to arrays)"""
    x = (1, 2, 3)
    y = (4, 5, 6)
    result = euclidean_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, np.sqrt(27))

def test_euclidean_dist_symmetry():
    """Test that distance is symmetric: d(x,y) = d(y,x)"""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    result1 = euclidean_dist(x, y)
    result2 = euclidean_dist(y, x)
    
    assert np.isclose(result1, result2)

#------------------------------
#------------------------------
## Manhattan Distance
#------------------------------
#------------------------------


#------------------------------
## Standard Tests
#------------------------------

def test_manhattan_dist_1d_basic():
    """Test basic 1D manhattan distance"""
    x = np.array([0, 0])
    y = np.array([3, 4])
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 7.0)

def test_manhattan_dist_2d_basic():
    """Test basic 2D manhattan distance with default axis"""
    x = np.array([[0, 0], [0, 1]])
    y = np.array([[1, 1], [1, 2]])
    result = manhattan_dist(x, y)
    
    assert result.shape == (2,) # type: ignore
    assert np.allclose(result, [2.0, 2.0])

def test_manhattan_dist_2d_axis_0():
    """Test 2D manhattan distance along axis 0"""
    x = np.array([[0, 0], [0, 1]])
    y = np.array([[1, 1], [1, 2]])
    result = manhattan_dist(x, y, axis=0)
    
    assert result.shape == (2,) # type: ignore
    assert np.allclose(result, [2.0, 2.0])

def test_manhattan_dist_2d_axis_1():
    """Test 2D manhattan distance along axis 1"""
    x = np.array([[0, 0], [0, 1]])
    y = np.array([[1, 1], [1, 2]])
    result = manhattan_dist(x, y, axis=1)
    
    assert result.shape == (2,)  # type: ignore
    assert np.allclose(result, [2.0, 2.0])

#------------------------------
## Zero Value Tests
#------------------------------

def test_manhattan_dist_identical_arrays_1d():
    """Test distance between identical 1D arrays (should be 0)"""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    result = manhattan_dist(x, y)
    
    assert np.isclose(result, 0.0)

def test_manhattan_dist_identical_arrays_2d():
    """Test distance between identical 2D arrays (should be 0)"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2], [3, 4]])
    result = manhattan_dist(x, y)
    
    assert np.allclose(result, [0.0, 0.0])

def test_manhattan_dist_all_zeros_1d():
    """Test distance between zero arrays"""
    x = np.array([0, 0, 0])
    y = np.array([0, 0, 0])
    result = manhattan_dist(x, y)
    
    assert np.isclose(result, 0.0)

def test_manhattan_dist_all_zeros_2d():
    """Test distance between zero 2D arrays"""
    x = np.array([[0, 0], [0, 0]])
    y = np.array([[0, 0], [0, 0]])
    result = manhattan_dist(x, y)
    
    assert np.allclose(result, [0.0, 0.0])

#------------------------------
## Constraint Tests: Scalar Inputs
#------------------------------

def test_manhattan_dist_scalar_fails():
    """Test that scalar raises ValueError"""
    x = np.array(5)
    y = np.array([1, 2])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 0 dimensions.\n2D validation error: Expected 2D. Got 0 dimensions."):
        manhattan_dist(y, x)

def test_manhattan_dist_both_scalars_fail():
    """Test that both scalars raise ValueError"""
    x = np.array(5)
    y = np.array(3)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 0 dimensions.\n2D validation error: Expected 2D. Got 0 dimensions."):
        manhattan_dist(x, y)

#------------------------------
## Constraint Tests: Dimensionality
#------------------------------

def test_manhattan_dist_1d_passes():
    """Test that 1D arrays pass dimensionality check"""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_manhattan_dist_2d_passes():
    """Test that 2D arrays pass dimensionality check"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    result = manhattan_dist(x, y)
    
    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_manhattan_dist_3d_fails():
    """Test that 3D arrays raise ValueError"""
    x = np.array([[[1, 2], [3, 4]]])
    y = np.array([[[5, 6], [7, 8]]])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 3 dimensions.\n2D validation error: Expected 2D. Got 3 dimensions."):
        manhattan_dist(x, y)

def test_manhattan_dist_4d_fails():
    """Test that 4D arrays raise ValueError"""
    x = np.ones((2, 2, 2, 2))
    y = np.ones((2, 2, 2, 2))
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 4 dimensions.\n2D validation error: Expected 2D. Got 4 dimensions."):
        manhattan_dist(x, y)

#------------------------------
## Constraint Tests: Empty Arrays
#------------------------------

def test_manhattan_dist_empty_fails():
    """Test that empty raises ValueError"""
    x = np.array([])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array cannot be empty.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)

    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array cannot be empty.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(y, x)

def test_manhattan_dist_both_empty_fail():
    """Test that both empty arrays raise ValueError"""
    x = np.array([])
    y = np.array([])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array cannot be empty.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)

def test_manhattan_dist_empty_2d_fails():
    """Test that empty 2D raises ValueError"""
    x = np.array([[]]).reshape(0, 2)
    y = np.array([[1, 2]])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 2 dimensions.\n2D validation error: Array cannot be empty."):
        manhattan_dist(x, y)

    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Expected 1D. Got 2 dimensions.\n2D validation error: Array cannot be empty."):
        manhattan_dist(y, x)

#------------------------------
## Constraint Tests: Shape Mismatch
#------------------------------

def test_manhattan_dist_different_lengths_1d_fails():
    """Test that 1D arrays with different lengths raise ValueError"""
    x = np.array([1, 2, 3])
    y = np.array([1, 2])
    
    with pytest.raises(ValueError, match=re.escape("x and y must have the same shape, got (3,) and (2,).")):
        manhattan_dist(x, y)

def test_manhattan_dist_different_shapes_2d_fails():
    """Test that 2D arrays with different shapes raise ValueError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    
    with pytest.raises(ValueError, match=re.escape("x and y must have the same shape, got (2, 2) and (2, 3).")):
        manhattan_dist(x, y)

def test_manhattan_dist_same_size_different_shape_fails():
    """Test that arrays with same size but different shape raise ValueError"""
    x = np.array([[1, 2, 3, 4]])
    y = np.array([[1, 2], [3, 4]])
    
    with pytest.raises(ValueError, match=re.escape("x and y must have the same shape, got (1, 4) and (2, 2).")):
        manhattan_dist(x, y)

def test_manhattan_dist_matching_shapes_passes():
    """Test that matching shapes pass constraint"""
    x = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[7, 8], [9, 10], [11, 12]])
    result = manhattan_dist(x, y)
    
    assert result.shape == (2,)  # type: ignore

#------------------------------
## Constraint Tests: Non-Numeric Values
#------------------------------

def test_manhattan_dist_string_fails():
    """Test that string values raise TypeError"""
    x = np.array(['a', 'b', 'c'])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must contain numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must contain numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(y, x)

def test_manhattan_dist_object_array_mixed_fails():
    """Test that object array with non-numeric raises TypeError"""
    x = np.array([1, 2, 'c'], dtype=object)
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array contains non-numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)

    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array contains non-numeric values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(y, x)

def test_manhattan_dist_numeric_object_array_passes():
    """Test that object array with only numeric values passes"""
    x = np.array([1, 2, 3], dtype=object)
    y = np.array([4, 5, 6], dtype=object)
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 9.0)

#------------------------------
## Constraint Tests: NaN and Infinite Values
#------------------------------

def test_manhattan_dist_nan_fails():
    """Test that NaN raises ValueError"""
    x = np.array([1, np.nan, 3])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(y, x)

def test_manhattan_dist_inf_fails():
    """Test that infinity raises a ValueError"""
    x = np.array([1, np.inf, 3])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(y, x)

def test_manhattan_dist_neg_inf_in_x_fails():
    """Test that negative infinity in x raises ValueError"""
    x = np.array([1, -np.inf, 3])
    y = np.array([1, 2, 3])
    
    with pytest.raises(ValueError, match="Input must be a 1D or 2D numeric array.\n\n1D validation error: Array must not contain NaN or infinite values.\n2D validation error: Expected 2D. Got 1 dimensions."):
        manhattan_dist(x, y)

def test_manhattan_dist_finite_values_pass():
    """Test that finite numeric values pass constraint"""
    x = np.array([1.5, -2.3, 0.0, 1e10, -1e10])
    y = np.array([2.5, -3.3, 1.0, 2e10, -2e10])
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isfinite(result)

#------------------------------
## Constraint Tests: Axis Parameter
#------------------------------

def test_manhattan_dist_axis_none():
    """Test axis=None with 1D and 2D arrays"""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    result = manhattan_dist(x, y, axis=None)
    
    assert isinstance(result, float)
    assert np.isclose(result, 9.0)

    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    result = manhattan_dist(x, y, axis=None)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_manhattan_dist_axis_negative_valid():
    """Test negative axis indexing (valid)"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    result = manhattan_dist(x, y, axis=-1)
    
    assert result.shape == (2,) # type: ignore
    assert np.allclose(result, [8.0, 8.0])

    result = manhattan_dist(x, y, axis=-2)
    assert result.shape == (2,) # type: ignore

def test_manhattan_dist_axis_out_of_bounds_fails():
    """Test axis out of bounds raises ValueError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    
    with pytest.raises(ValueError, match="'axis'=2 out of bounds"):
        manhattan_dist(x, y, axis=2)
    
    with pytest.raises(ValueError, match="'axis'=-3 out of bounds"):
        manhattan_dist(x, y, axis=-3)

def test_manhattan_dist_axis_non_integer_fails():
    """Test non-integer axis raises TypeError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    
    with pytest.raises(TypeError, match="'axis' must be an integer or None"):
        manhattan_dist(x, y, axis=1.5) # type: ignore

def test_manhattan_dist_axis_string_fails():
    """Test string axis raises TypeError"""
    x = np.array([[1, 2], [3, 4]])
    y = np.array([[5, 6], [7, 8]])
    
    with pytest.raises(TypeError, match="'axis' must be an integer or None"):
        manhattan_dist(x, y, axis='0') # type: ignore

#------------------------------
## Edge Case Tests
#------------------------------

def test_manhattan_dist_single_element_1d():
    """Test 1D arrays with single element"""
    x = np.array([5.0])
    y = np.array([2.0])
    result = manhattan_dist(x, y)
    
    assert np.isclose(result, 3.0)

def test_manhattan_dist_single_element_2d():
    """Test 2D arrays with single element"""
    x = np.array([[5.0]])
    y = np.array([[2.0]])
    result = manhattan_dist(x, y)
    
    assert result.shape == (1,) # type: ignore
    assert np.isclose(result[0], 3.0) # type: ignore

def test_manhattan_dist_large_values():
    """Test with very large values"""
    x = np.array([1e100, 1e100])
    y = np.array([2e100, 2e100])
    result = manhattan_dist(x, y)
    
    assert np.isfinite(result)
    assert result > 0

def test_manhattan_dist_very_small_values():
    """Test with very small values"""
    x = np.array([1e-100, 1e-100])
    y = np.array([2e-100, 2e-100])
    result = manhattan_dist(x, y)
    
    assert np.isfinite(result)
    assert result >= 0

def test_manhattan_dist_negative_values():
    """Test with negative values"""
    x = np.array([-5, -10, -15])
    y = np.array([-2, -8, -12])
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_manhattan_dist_mixed_signs():
    """Test with mixed positive and negative values"""
    x = np.array([-5, 10, -15])
    y = np.array([2, -8, 12])
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_manhattan_dist_integer_arrays():
    """Test with integer arrays"""
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4, 5, 6], dtype=np.int32)
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 9.0)

def test_manhattan_dist_float_arrays():
    """Test with float arrays"""
    x = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    y = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 9.0)

def test_manhattan_dist_mixed_dtypes():
    """Test with mixed integer and float arrays"""
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([4.5, 5.5, 6.5], dtype=np.float64)
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert result > 0

def test_manhattan_dist_list_inputs():
    """Test with list inputs (should be converted to arrays)"""
    x = [1, 2, 3]
    y = [4, 5, 6]
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 9.0)

def test_manhattan_dist_tuple_inputs():
    """Test with tuple inputs (should be converted to arrays)"""
    x = (1, 2, 3)
    y = (4, 5, 6)
    result = manhattan_dist(x, y)
    
    assert isinstance(result, float)
    assert np.isclose(result, 9.0)

def test_manhattan_dist_symmetry():
    """Test that distance is symmetric: d(x,y) = d(y,x)"""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    result1 = manhattan_dist(x, y)
    result2 = manhattan_dist(y, x)
    
    assert np.isclose(result1, result2)