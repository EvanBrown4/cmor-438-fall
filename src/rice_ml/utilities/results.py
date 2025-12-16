import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    'confusion_matrix',
    'plot_confusion_matrix',
]


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int | None = None
) -> np.ndarray:
    """
    Compute a confusion matrix for classification.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground-truth class labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels.
    num_classes : int, optional
        Number of classes. If None, inferred from data.

    Returns
    -------
    cm : ndarray of shape (num_classes, num_classes)
        Confusion matrix where cm[i, j] is the number of times
        class i was predicted as class j.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    # Ensure labels are integer-valued (allow floats like 0.0, 1.0)
    if not np.all(np.equal(y_true, np.floor(y_true))):
        raise ValueError("y_true must contain integer class labels.")
    if not np.all(np.equal(y_pred, np.floor(y_pred))):
        raise ValueError("y_pred must contain integer class labels.")

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    max_label = int(max(y_true.max(), y_pred.max()))

    if num_classes is None:
        num_classes = max_label + 1
    else:
        if num_classes <= max_label:
            raise ValueError(
                "num_classes must be greater than the maximum class label."
            )

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    return cm


def plot_confusion_matrix(cm, class_labels=None, normalize=False):
    """
    Plot a confusion matrix using matplotlib.

    Parameters
    ----------
    cm : ndarray of shape (n_classes, n_classes)
        Confusion matrix where cm[i, j] is the number of samples
        with true label i predicted as label j.
    class_labels : list or array-like, optional
        Labels to display on the x- and y-axes.
    normalize : bool, default=False
        If True, normalize each row of the confusion matrix.
    """
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (Normalized)" if normalize else ""))

    if class_labels is not None:
        ax.set_xticks(range(len(class_labels)))
        ax.set_yticks(range(len(class_labels)))
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = f"{value:.2f}" if normalize else str(int(value))

            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if value > cm.max() * 0.5 else "black"
            )

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()
