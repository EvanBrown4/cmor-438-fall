import numpy as np
import pandas as pd
import pytest
from typing import Union

from src.utilities.postprocess import *

#------------------------------
## majority_label Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_majority_label_basic():
    """Test basic majority label selection"""
    labels = np.array(['a', 'b', 'b', 'c'])
    result = majority_label(labels)
    
    assert result == 'b'

def test_majority_label_numeric():
    """Test with numeric labels"""
    labels = np.array([1, 2, 2, 3, 2])
    result = majority_label(labels)
    
    assert result == 2

def test_majority_label_all_same():
    """Test when all labels are the same"""
    labels = np.array([5, 5, 5, 5])
    result = majority_label(labels)
    
    assert result == 5

def test_majority_label_tie_returns_min():
    """Test that ties are broken by returning minimum"""
    labels = np.array([1, 2, 1, 2])
    result = majority_label(labels)
    
    assert result == 1

def test_majority_label_three_way_tie():
    """Test three-way tie returns minimum"""
    labels = np.array([3, 1, 2, 3, 1, 2])
    result = majority_label(labels)
    
    assert result == 1

#------------------------------
## Edge Cases
#------------------------------

def test_majority_label_single_element():
    """Test with single element"""
    labels = np.array([42])
    result = majority_label(labels)
    
    assert result == 42

def test_majority_label_two_elements_same():
    """Test with two identical elements"""
    labels = np.array([7, 7])
    result = majority_label(labels)
    
    assert result == 7

def test_majority_label_two_elements_different():
    """Test with two different elements"""
    labels = np.array([5, 3])
    result = majority_label(labels)
    
    assert result == 3

def test_majority_label_floats():
    """Test with float labels"""
    labels = np.array([1.5, 2.5, 1.5, 3.5])
    result = majority_label(labels)
    
    assert result == 1.5

def test_majority_label_negative_numbers():
    """Test with negative numbers"""
    labels = np.array([-1, -2, -1, -3])
    result = majority_label(labels)
    
    assert result == -1

def test_majority_label_mixed_signs():
    """Test with mixed positive and negative"""
    labels = np.array([-1, 1, -1, 2])
    result = majority_label(labels)
    
    assert result == -1

def test_majority_label_list_input():
    """Test with list input"""
    labels = [1, 2, 2, 3]
    result = majority_label(labels)
    
    assert result == 2

def test_majority_label_tuple_input():
    """Test with tuple input"""
    labels = (1, 2, 2, 3)
    result = majority_label(labels)
    
    assert result == 2

def test_majority_label_pandas_series():
    """Test with pandas Series"""
    labels = pd.Series([1, 2, 2, 3])
    result = majority_label(labels)
    
    assert result == 2

def test_majority_label_large_array():
    """Test with large array"""
    labels = np.array([1] * 100 + [2] * 150 + [3] * 75)
    result = majority_label(labels)
    
    assert result == 2

def test_majority_label_zeros():
    """Test with zeros"""
    labels = np.array([0, 1, 0, 2, 0])
    result = majority_label(labels)
    
    assert result == 0

def test_majority_label_mixed_types_object():
    """Test that mixed types in object array fail"""
    labels = np.array([1, 2, 'a'], dtype=object)
    
    with pytest.raises(TypeError, match="All labels must have the same Python type."):
        majority_label(labels)

def test_majority_label_nan_fails():
    """Test that NaN values fail"""
    labels = np.array([1, 2, np.nan, 2])
    
    with pytest.raises(ValueError, match="Array must not contain NaN or infinite values"):
        majority_label(labels)

def test_majority_label_inf_fails():
    """Test that infinite values fail"""
    labels = np.array([1, 2, np.inf, 2])
    
    with pytest.raises(ValueError, match="Array must not contain NaN or infinite values"):
        majority_label(labels)

#------------------------------
## average_label Tests
#------------------------------

#------------------------------
## Standard Tests
#------------------------------

def test_average_label_basic():
    """Test basic average label calculation"""
    labels = np.array([1, 2, 3, 4])
    dists = np.array([1, 1, 1, 1])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 2.5)

def test_average_label_weighted():
    """Test weighted average with different distances"""
    labels = np.array([1.0, 2.0, 3.0])
    dists = np.array([1.0, 2.0, 3.0])
    result = average_label(labels, dists)
    
    # weights: 1/1=1, 1/2=0.5, 1/3=0.333
    # weighted sum: 1*1 + 2*0.5 + 3*0.333 = 1 + 1 + 1 = 3
    # sum of weights: 1 + 0.5 + 0.333 = 1.833
    expected = (1*1 + 2*0.5 + 3*(1/3)) / (1 + 0.5 + 1/3)
    assert np.isclose(result, expected)

def test_average_label_closer_weighted_more():
    """Test that closer points (smaller distances) are weighted more"""
    labels = np.array([1.0, 10.0])
    dists = np.array([1.0, 10.0])
    result = average_label(labels, dists)
    
    # weight for 1.0: 1/1 = 1
    # weight for 10.0: 1/10 = 0.1
    # result should be closer to 1.0 than 10.0
    assert result < 5.5

def test_average_label_all_same():
    """Test when all labels are the same"""
    labels = np.array([5.0, 5.0, 5.0])
    dists = np.array([1.0, 2.0, 3.0])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 5.0)

def test_average_label_single_element():
    """Test with single element"""
    labels = np.array([42.0])
    dists = np.array([1.0])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 42.0)

#------------------------------
## Edge Cases
#------------------------------

def test_average_label_two_elements():
    """Test with two elements"""
    labels = np.array([1.0, 3.0])
    dists = np.array([1.0, 1.0])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 2.0)

def test_average_label_zero_distance_returns_exact_label():
    labels = np.array([1, 2, 3])
    dists = np.array([0, 1, 2])

    result = average_label(labels, dists)
    assert result == 1

def test_average_label_negative_labels():
    """Test with negative labels"""
    labels = np.array([-1.0, -2.0, -3.0])
    dists = np.array([1.0, 1.0, 1.0])
    result = average_label(labels, dists)
    
    assert np.isclose(result, -2.0)

def test_average_label_mixed_sign_labels():
    """Test with mixed sign labels"""
    labels = np.array([-1.0, 1.0])
    dists = np.array([1.0, 1.0])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 0.0)

def test_average_label_large_distances():
    """Test with large distance values"""
    labels = np.array([1.0, 2.0, 3.0])
    dists = np.array([1e10, 1e10, 1e10])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 2.0)

def test_average_label_small_distances():
    """Test with very small distance values"""
    labels = np.array([1.0, 2.0, 3.0])
    dists = np.array([1e-10, 1e-10, 1e-10])
    result = average_label(labels, dists)
    
    assert np.isclose(result, 2.0)

def test_average_label_list_inputs():
    """Test with list inputs"""
    labels = [1, 2, 3]
    dists = [1, 1, 1]
    result = average_label(labels, dists) # type: ignore
    
    assert np.isclose(result, 2.0)

def test_average_label_tuple_inputs():
    """Test with tuple inputs"""
    labels = (1, 2, 3)
    dists = (1, 1, 1)
    result = average_label(labels, dists) # type: ignore
    
    assert np.isclose(result, 2.0)

def test_average_label_pandas_series():
    """Test with pandas Series"""
    labels = pd.Series([1.0, 2.0, 3.0])
    dists = pd.Series([1.0, 1.0, 1.0])
    result = average_label(labels, dists) # type: ignore
    
    assert np.isclose(result, 2.0)

def test_average_label_string_labels_fail():
    """Test that string labels fail"""
    labels = np.array(['a', 'b', 'c'])
    dists = np.array([1, 1, 1])
    
    with pytest.raises(TypeError, match="Array must contain numeric values"):
        average_label(labels, dists)

def test_average_label_string_dists_fail():
    """Test that string distances fail"""
    labels = np.array([1, 2, 3])
    dists = np.array(['a', 'b', 'c'])
    
    with pytest.raises(TypeError, match="Array must contain numeric values"):
        average_label(labels, dists)

def test_average_label_nan_in_labels_fails():
    """Test that NaN in labels fails"""
    labels = np.array([1, np.nan, 3])
    dists = np.array([1, 1, 1])
    
    with pytest.raises(ValueError, match="Array must not contain NaN or infinite values"):
        average_label(labels, dists)

def test_average_label_nan_in_dists_fails():
    """Test that NaN in distances fails"""
    labels = np.array([1, 2, 3])
    dists = np.array([1, np.nan, 1])
    
    with pytest.raises(ValueError, match="Array must not contain NaN or infinite values"):
        average_label(labels, dists)

def test_average_label_inf_in_labels_fails():
    """Test that infinity in labels fails"""
    labels = np.array([1, np.inf, 3])
    dists = np.array([1, 1, 1])
    
    with pytest.raises(ValueError, match="Array must not contain NaN or infinite values"):
        average_label(labels, dists)

def test_average_label_inf_in_dists_fails():
    """Test that infinity in distances fails"""
    labels = np.array([1, 2, 3])
    dists = np.array([1, np.inf, 1])
    
    with pytest.raises(ValueError, match="Array must not contain NaN or infinite values"):
        average_label(labels, dists)