import numpy as np

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
    confusion_matrix = np.array([[true_negatives, false_positives],
                                 [false_negatives, true_positives]])

    return confusion_matrix


def calculate_fdr(W_true:np.ndarray, W_pred:np.ndarray) -> float:
    Wt = W_true.T
    R = np.sum((Wt == 1) & (W_pred == 1))  # Reversed edges
    FP = np.sum((W_true == 0) & (W_pred == 1))  # False positives
    P = np.sum(W_pred == 1)  # Total positive edges
    return (R + FP) / P

def calculate_tpr(W_true:np.ndarray, W_pred:np.ndarray) -> float:
    TP = np.sum((W_true == 1) & (W_pred == 1))  # True positives
    T = np.sum(W_true == 1)  # Total true edges
    return TP / T

def calculate_fpr(W_true:np.ndarray, W_pred:np.ndarray) -> float:
    R = np.sum((W_true == 1) & (W_pred == -1))  # Reversed edges
    FP = np.sum((W_true == 0) & (W_pred == 1))  # False positives
    F = np.sum(W_true == 0)  # Total non-edges in ground truth
    return (R + FP) / F

def calculate_shd(W_true:np.ndarray, W_pred:np.ndarray) -> int:
    Wt = W_true.T
    E = np.sum((W_true == 0) & (W_pred == 1))  # Extra edges
    M = np.sum((W_true == 1) & (W_pred == 0))  # Missing edges
    R = np.sum((Wt == 1) & (W_pred == 1))  # Reversed edges
    return E + M + R

def roc_auc(W_true, W_pred, num_thresholds=1000):
    thresholds = np.linspace(0, 1, num_thresholds)
    tpr_values = []
    fpr_values = []
    for threshold in thresholds:
        tpr_values.append(
            calculate_tpr(W_true, (W_pred >= threshold).astype(int))
        )
        fpr_values.append(
            calculate_fpr(W_true, (W_pred >= threshold).astype(int))
        )
    tpr_array = np.array(tpr_values)
    fpr_array = np.array(fpr_values)

    # Calculate AUC using trapezoidal rule
    auc = np.trapz(tpr_array, fpr_array)

    return tpr_array, fpr_array, auc