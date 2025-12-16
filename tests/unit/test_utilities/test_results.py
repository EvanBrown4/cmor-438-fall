import pytest
import numpy as np
from src.rice_ml.utilities.results import confusion_matrix


#------------------------------
## Standard Tests
#------------------------------

def test_confusion_matrix_basic():
    """Test basic confusion matrix computation"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # True negatives
    assert cm[1, 1] == 2  # True positives


def test_confusion_matrix_with_errors():
    """Test confusion matrix with misclassifications"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm[0, 0] == 1  # True negative
    assert cm[0, 1] == 1  # False positive
    assert cm[1, 0] == 1  # False negative
    assert cm[1, 1] == 1  # True positive


def test_confusion_matrix_multiclass():
    """Test confusion matrix for multiclass classification"""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[2, 2] == 2


def test_confusion_matrix_multiclass_errors():
    """Test multiclass confusion matrix with errors"""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 1, 0, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1  # Class 0 correctly predicted
    assert cm[0, 1] == 1  # Class 0 predicted as 1
    assert cm[1, 2] == 1  # Class 1 predicted as 2
    assert cm[1, 0] == 1  # Class 1 predicted as 0
    assert cm[2, 1] == 1  # Class 2 predicted as 1
    assert cm[2, 2] == 1  # Class 2 correctly predicted


def test_confusion_matrix_explicit_num_classes():
    """Test confusion matrix with explicit num_classes"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    
    cm = confusion_matrix(y_true, y_pred, num_classes=3)
    
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2
    assert cm[1, 1] == 2
    assert cm[2, 2] == 0  # No samples of class 2


def test_confusion_matrix_inferred_classes():
    """Test that num_classes is inferred correctly"""
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (4, 4)


def test_confusion_matrix_sum_equals_samples():
    """Test that confusion matrix sum equals number of samples"""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    y_pred = np.array([0, 2, 1, 1, 0, 2, 0, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.sum() == len(y_true)


def test_confusion_matrix_diagonal_perfect():
    """Test perfect predictions result in diagonal matrix"""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    
    # All values should be on diagonal
    for i in range(3):
        for j in range(3):
            if i == j:
                assert cm[i, j] > 0
            else:
                assert cm[i, j] == 0


def test_confusion_matrix_row_sums():
    """Test row sums equal class counts"""
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm[0, :].sum() == 3  # Class 0 appears 3 times
    assert cm[1, :].sum() == 2  # Class 1 appears 2 times
    assert cm[2, :].sum() == 1  # Class 2 appears 1 time


def test_confusion_matrix_dtype_int():
    """Test confusion matrix has integer dtype"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.dtype == int


def test_confusion_matrix_large_dataset():
    """Test confusion matrix with larger dataset"""
    np.random.seed(42)
    y_true = np.random.randint(0, 5, size=1000)
    y_pred = np.random.randint(0, 5, size=1000)
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (5, 5)
    assert cm.sum() == 1000


def test_confusion_matrix_all_same_class():
    """Test confusion matrix when all samples are same class"""
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm[1, 1] == 4
    assert cm.sum() == 4


def test_confusion_matrix_all_misclassified():
    """Test confusion matrix when all predictions are wrong"""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm[0, 0] == 0  # No correct predictions
    assert cm[0, 1] == 4  # All misclassified as class 1


def test_confusion_matrix_list_input():
    """Test confusion matrix accepts lists"""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 0]
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (2, 2)
    assert isinstance(cm, np.ndarray)


def test_confusion_matrix_float_labels():
    """Test confusion matrix with float labels that are integers"""
    y_true = np.array([0.0, 1.0, 2.0, 0.0])
    y_pred = np.array([0.0, 1.0, 2.0, 1.0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 1
    assert cm[0, 1] == 1


#------------------------------
## Error Handling Tests
#------------------------------

def test_confusion_matrix_shape_mismatch():
    """Test confusion matrix raises error for mismatched shapes"""
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1])
    
    with pytest.raises(ValueError, match="must have the same shape"):
        confusion_matrix(y_true, y_pred)


def test_confusion_matrix_negative_num_classes():
    """Invalid num_classes should raise ValueError"""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])

    with pytest.raises(ValueError):
        confusion_matrix(y_true, y_pred, num_classes=1)


#------------------------------
## Edge Cases
#------------------------------

def test_confusion_matrix_single_sample():
    """Test confusion matrix with single sample"""
    y_true = np.array([0])
    y_pred = np.array([0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (1, 1)
    assert cm[0, 0] == 1


def test_confusion_matrix_two_samples():
    """Test confusion matrix with two samples"""
    y_true = np.array([0, 1])
    y_pred = np.array([1, 0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (2, 2)
    assert cm[0, 1] == 1
    assert cm[1, 0] == 1


def test_confusion_matrix_empty_classes():
    """Test confusion matrix with unused classes"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred, num_classes=5)
    
    assert cm.shape == (5, 5)
    assert cm[2:, :].sum() == 0  # Classes 2-4 unused
    assert cm[:, 2:].sum() == 0


def test_confusion_matrix_binary_perfect():
    """Test binary classification with perfect accuracy"""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm[0, 0] == 3  # True negatives
    assert cm[0, 1] == 0  # False positives
    assert cm[1, 0] == 0  # False negatives
    assert cm[1, 1] == 3  # True positives


def test_confusion_matrix_binary_all_wrong():
    """Test binary classification with all wrong predictions"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm[0, 0] == 0  # No true negatives
    assert cm[0, 1] == 2  # All class 0 predicted as 1
    assert cm[1, 0] == 2  # All class 1 predicted as 0
    assert cm[1, 1] == 0  # No true positives


def test_confusion_matrix_noncontiguous_classes():
    """Test confusion matrix with non-contiguous class labels"""
    y_true = np.array([0, 2, 0, 2])
    y_pred = np.array([0, 2, 0, 2])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (3, 3)
    assert cm[0, 0] == 2
    assert cm[2, 2] == 2
    assert cm[1, 1] == 0  # Class 1 never used


def test_confusion_matrix_high_class_numbers():
    """Test confusion matrix with high class numbers"""
    y_true = np.array([10, 20, 10, 20])
    y_pred = np.array([10, 20, 10, 20])
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (21, 21)
    assert cm[10, 10] == 2
    assert cm[20, 20] == 2


def test_confusion_matrix_zeros_matrix():
    """Test confusion matrix when predictions don't match any true labels"""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    
    cm = confusion_matrix(y_true, y_pred, num_classes=2)
    
    assert cm[0, 0] == 0
    assert cm[0, 1] == 4
    assert cm[1, 0] == 0
    assert cm[1, 1] == 0


def test_confusion_matrix_many_classes():
    """Test confusion matrix with many classes"""
    np.random.seed(42)
    y_true = np.random.randint(0, 10, size=100)
    y_pred = np.random.randint(0, 10, size=100)
    
    cm = confusion_matrix(y_true, y_pred)
    
    assert cm.shape == (10, 10)
    assert cm.sum() == 100