import pytest
import numpy as np
import pandas as pd
from src.utilities.preprocess import train_test_split, normalize, scale

def test_tts_standard_x_and_y():
    """Test with standard numpy arrays"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(y_train) + len(y_test) == len(y)
    assert len(X_test) == 1  # 20% of 5 = 1
    assert len(y_test) == 1

def test_tts_no_y_provided():
    """Test with only X provided"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_test) == 1

def test_tts_multiple_standard_splits():
    """Test multiple standard cases"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    assert len(X_train) == 35
    assert len(X_test) == 15
    assert len(y_train) == 35
    assert len(y_test) == 15

def test_tts_list_x_and_y():
    """Test with lists"""
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = [0, 1, 0, 1, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert len(X_train) + len(X_test) == len(X)

def test_tts_tuple_x_and_y():
    """Test with tuples"""
    X = tuple([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = tuple([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)

def test_tts_dataframe_x_and_series_y():
    """Test with pandas DataFrame and Series"""
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
    y = pd.Series([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert len(X_train) + len(X_test) == len(X)

def test_tts_numpy_x_list_y():
    """Test with numpy array X and list y"""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = [0, 1, 0, 1, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X)

def test_tts_list_x_numpy_y():
    """Test with list X and numpy array y"""
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = np.array([0, 1, 0, 1, 0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X)

def test_tts_dataframe_x_list_y():
    """Test with DataFrame X and list y"""
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [6, 7, 8, 9, 10]})
    y = [0, 1, 0, 1, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_train) + len(X_test) == len(X)


## Shuffle and Random State Tests
def test_tts_shuffle_true():
    """Test with shuffle=True"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    
    # Check that the data is actually shuffled
    assert not np.array_equal(y_train, np.arange(40))

def test_tts_shuffle_false():
    """Test with shuffle=False"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # With shuffle=False, training data should be first 40 samples
    assert np.array_equal(y_train, np.arange(40))
    assert np.array_equal(y_test, np.arange(40, 50))

def test_tts_random_state_consistency():
    """Test that random_state produces consistent results"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert np.array_equal(X_train1, X_train2)
    assert np.array_equal(X_test1, X_test2)
    assert np.array_equal(y_train1, y_train2)
    assert np.array_equal(y_test1, y_test2)

def test_tts_no_random_state():
    """Test without random_state (should produce different results)"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, shuffle=True)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    # Should have different ordering (with very high probability)
    assert not np.array_equal(y_train1, y_train2) or not np.array_equal(y_test1, y_test2)

def test_tts_different_random_states():
    """Test that different random states produce different results"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=123)
    
    assert not np.array_equal(y_train1, y_train2)

## Stratify Tests
def test_tts_stratify_basic():
    """Test basic stratified split"""
    X = np.arange(100).reshape(50, 2)
    y = np.array([0] * 25 + [1] * 25)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Check that class proportions are maintained
    train_class_0 = np.sum(y_train == 0)
    train_class_1 = np.sum(y_train == 1)
    test_class_0 = np.sum(y_test == 0)
    test_class_1 = np.sum(y_test == 1)
    
    # Should maintain approximately 50-50 split
    assert abs(train_class_0 - train_class_1) <= 1
    assert abs(test_class_0 - test_class_1) <= 1

def test_tts_stratify_multiclass():
    """Test stratified split with multiple classes"""
    X = np.arange(150).reshape(50, 3)
    y = np.array([0] * 20 + [1] * 20 + [2] * 10)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Check that proportions are roughly maintained
    train_props = [np.sum(y_train == i) / len(y_train) for i in range(3)]
    test_props = [np.sum(y_test == i) / len(y_test) for i in range(3)]
    original_props = [np.sum(y == i) / len(y) for i in range(3)]
    
    # All proportions should be close to original
    for i in range(3):
        assert abs(train_props[i] - original_props[i]) < 0.1
        assert abs(test_props[i] - original_props[i]) < 0.1

## Test Size Tests
def test_tts_test_size_0_5():
    """Test with 50-50 split"""
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    assert len(X_train) == 50
    assert len(X_test) == 50

def test_tts_test_size_0_1():
    """Test with 10% test size"""
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    assert len(X_test) == 10
    assert len(X_train) == 90

def test_tts_test_size_0_9():
    """Test with 90% test size"""
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=42)
    
    assert len(X_test) == 90
    assert len(X_train) == 10

def test_tts_test_size_very_small():
    """Test with very small test size"""
    X = np.arange(200).reshape(100, 2)
    y = np.arange(100)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
    
    assert len(X_test) == 1  # Ceiling of 100 * 0.01 = 1
    assert len(X_train) == 99

## Invalid Inputs Studying
def test_tts_mismatched_x_y_lengths():
    """Test that mismatched X and y lengths raise an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1])  # Length mismatch
    
    with pytest.raises((ValueError, AssertionError)):
        train_test_split(X, y, test_size=0.2)

def test_tts_invalid_test_size_zero():
    """Test that test_size=0 raises an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        train_test_split(X, y, test_size=0.0)

def test_tts_invalid_test_size_one():
    """Test that test_size=1.0 raises an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        train_test_split(X, y, test_size=1.0)

def test_tts_invalid_test_size_negative(self):
    """Test that negative test_size raises an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        train_test_split(X, y, test_size=-0.1)

def test_tts_invalid_test_size_greater_than_one():
    """Test that test_size > 1.0 raises an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        train_test_split(X, y, test_size=1.5)

def test_tts_invalid_random_state_negative():
    """Test that negative random_state raises an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    with pytest.raises((ValueError, AssertionError)):
        train_test_split(X, y, test_size=0.2, random_state=-1)

def test_tts_invalid_random_state_too_large():
    """Test that random_state >= 2^32 raises an error"""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    
    with pytest.raises((ValueError, AssertionError, OverflowError)):
        train_test_split(X, y, test_size=0.2, random_state=2**32)

def test_tts_size_one_x_and_y():
    """Test with single sample"""
    X = np.array([[1, 2]])
    y = np.array([0])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # With ceiling, 1 * 0.2 = 0.2 -> ceiling to 1
    assert len(X_test) == 1
    assert len(X_train) == 0

def test_tts_size_zero_x_and_y():
    """Test with empty arrays"""
    X = np.array([]).reshape(0, 2)
    y = np.array([])
    
    # Should either handle gracefully or raise an error
    with pytest.raises((ValueError, AssertionError, IndexError)):
        train_test_split(X, y, test_size=0.2)

def test_tts_ceiling_behavior():
    """Test that ceiling is applied correctly when test_size would result in fractional samples"""
    X = np.arange(6).reshape(3, 2)
    y = np.array([0, 1, 2])
    
    # 3 * 0.2 = 0.6, should ceiling to 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    assert len(X_test) == 1
    assert len(X_train) == 2

def test_tts_size_two_with_fifty_percent_split():
    """Test with 2 samples and 50% split"""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    assert len(X_test) == 1
    assert len(X_train) == 1

def test_tts_very_small_test_size_resulting_in_zero_without_ceiling():
    """Test case where test_size * n < 1 (would be 0 without ceiling)"""
    X = np.arange(10).reshape(5, 2)
    y = np.array([0, 1, 2, 3, 4])
    
    # 5 * 0.1 = 0.5, without ceiling would be 0, with ceiling should be 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    assert len(X_test) == 1  # Ceiling of 0.5
    assert len(X_train) == 4

def test_tts_extreme_small_test_size():
    """Test with extremely small test_size"""
    X = np.arange(2000).reshape(1000, 2)
    y = np.arange(1000)
    
    # 1000 * 0.001 = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=42)
    
    assert len(X_test) == 1
    assert len(X_train) == 999

def test_tts_no_data_loss():
    """Test that all samples are preserved (no loss or duplication)"""
    X = np.arange(100).reshape(50, 2)
    y = np.arange(50)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Combine and check all samples are present
    all_y = np.concatenate([y_train, y_test])
    assert len(all_y) == len(y)
    assert set(all_y) == set(y)

def test_tts_x_y_correspondence():
    """Test that X and y correspondence is maintained"""
    X = np.arange(100).reshape(50, 2)
    y = X[:, 0]  # y is first column of X
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Check correspondence is maintained
    assert np.array_equal(X_train[:, 0], y_train)
    assert np.array_equal(X_test[:, 0], y_test)