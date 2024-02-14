from typing import Tuple
import numpy as np
from scipy.linalg import expm

def is_dag(W:np.ndarray) -> bool:
    d = W.shape[0]
    return (np.trace(expm(W*W)) - d) <= 1e-6

def find_dag_violation_threshold(W:np.ndarray, thresholds:np.ndarray=None) -> float:
    W_abs = np.abs(W)
    if not thresholds:
        thresholds = np.sort(W_abs.flatten())
    left, right = 0, len(thresholds) - 1
    violation_threshold = 0
    # Binary serach to find threshold
    while left <= right:
        mid = (left + right) // 2
        threshold = thresholds[mid]
        # Apply the threshold
        W_thresholded = np.where(W_abs > threshold, 1, 0)
        if not is_dag(W_thresholded):
            left = mid + 1  # Move right if it is not a DAG
        else:
            violation_threshold = threshold  # Update violation threshold
            right = mid - 1  # Move left if is DAG
    return violation_threshold

def confusion_matrix(W_true:np.ndarray, W_pred:np.ndarray) -> np.ndarray:
    """Calculates the confusion matrix given the true causal graph and the found one

    Args:
        W_true (np.ndarray): True causal structure with d*d dimensions
        W_pred (np.ndarray): Predicted causal structure with d"d dimensions

    Returns:
        np.ndarray: 2 by 2 matrix consisting of `[[TP, FP], [FN, TN]]`
    """
    true_positives = np.sum(
        np.logical_and(W_true == 1, W_pred == 1)
    )
    true_negatives = np.sum(
        np.logical_and(W_true == 0, W_pred == 0)
    )
    false_positives = np.sum(
        np.logical_and(W_true == 0, W_pred == 1)
    )
    false_negatives = np.sum(
        np.logical_and(W_true == 1, W_pred == 0)
    )
    confusion_matrix = np.array([[true_positives, false_positives],
                                 [false_negatives, true_negatives]])

    return confusion_matrix

def calculate_fdr(W_true:np.ndarray, W_pred:np.ndarray) -> float:
    Wt = W_true.T
    R = np.sum((Wt == 1) & (W_pred == 1))  # Reversed edges
    FP = np.sum((W_true == 0) & (W_pred == 1))  # False positives
    P = np.sum(W_pred == 1)  # Total positive edges
    return (R + FP) / P if P != 0 else 0

def calculate_tpr(W_true:np.ndarray, W_pred:np.ndarray) -> float:
    TP = np.sum((W_true == 1) & (W_pred == 1))  # True positives
    T = np.sum(W_true == 1)  # Total true edges
    return TP / T if T != 0 else 0

def calculate_fpr(W_true:np.ndarray, W_pred:np.ndarray) -> float:
    R = np.sum((W_true == 1) & (W_pred == -1))  # Reversed edges
    FP = np.sum((W_true == 0) & (W_pred == 1))  # False positives
    F = np.sum(W_true == 0)  # Total non-edges in ground truth
    return (R + FP) / F if F != 0 else 0

def calculate_shd(W_true:np.ndarray, W_pred:np.ndarray) -> int:
    Wt = W_true.T
    E = np.sum((W_true == 0) & (W_pred == 1))  # Extra edges
    M = np.sum((W_true == 1) & (W_pred == 0))  # Missing edges
    R = np.sum((Wt == 1) & (W_pred == 1))  # Reversed edges
    return E + M + R

def roc_auc(W_true, W_pred, num_thresholds=100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    thresholds = np.linspace(0, np.max(np.abs(W_pred)), num_thresholds, endpoint=True)
    tpr_values = []
    fpr_values = []
    for threshold in thresholds:
        tpr_values.append(
            calculate_tpr(W_true, (np.abs(W_pred) >= threshold).astype(int))
        )
        fpr_values.append(
            calculate_fpr(W_true, (np.abs(W_pred) >= threshold).astype(int))
        )
    tpr_array = np.array(tpr_values)
    fpr_array = np.array(fpr_values)

    # Calculate AUC using trapezoidal rule
    auc = -np.trapz(y=tpr_array, x=fpr_array) # "-" because of flipped array

    return tpr_array, fpr_array, thresholds, auc