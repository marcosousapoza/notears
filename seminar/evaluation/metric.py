import numpy as np


def get_confusion_matrix(dag: np.ndarray, dag_found: np.ndarray) -> np.ndarray:
    # Calculate the findings
    dag, dag_found = dag.astype(bool), dag_found.astype(bool)
    true_positives = np.sum(dag & dag_found)
    false_positives = np.sum(~dag & dag_found)
    false_negatives = np.sum(dag & ~dag_found)
    true_negatives = np.sum(~dag & ~dag_found)

    # Create confusion matrix
    conf_matrix = np.array([[true_positives, false_positives],
                            [false_negatives, true_negatives]])

    return conf_matrix