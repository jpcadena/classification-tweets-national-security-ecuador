"""
Evaluation script including improvement and tests.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(y_pred: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """
    Evaluate a binary classification ml_model based on several metrics.
    :param y_pred: Predicted binary labels
    :type y_pred: np.ndarray
    :param y_test: True binary values'
    :type y_test: np.ndarray
    :return: The confusion matrix
    :rtype: np.ndarray
    """
    accuracy: float = accuracy_score(y_test, y_pred)
    precision: float = precision_score(y_test, y_pred)
    recall: float = recall_score(y_test, y_pred)
    f1_s: float = f1_score(y_test, y_pred)
    roc_auc: float = roc_auc_score(y_test, y_pred)
    conf_matrix: np.ndarray = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1_s)
    print("ROC AUC:", roc_auc)
    print("Confusion Matrix:")
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    return conf_matrix
