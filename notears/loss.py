import numpy as np

def linear_sem_loss(X, W):
    """
    Calculate the linear structural equation modeling (SEM) loss and its gradient.

    Parameters:
    - X (numpy.ndarray): Input data matrix.
    - W (numpy.ndarray): Structural coefficients matrix.

    Returns:
    - Tuple[float, numpy.ndarray]: Loss and gradient.
    """
    assert X.shape[1] == W.shape[0], 'Incompatible dimensions between X and W'

    # get dimension of matricies
    n, d = X.shape
    # loss calculation
    loss = (0.5 / n) * np.linalg.norm(X - X.dot(W), ord='fro')**2
    # gradient calculation
    d_loss = (1 / n) * X.T @ X @ (W-np.identity(d))

    return loss, d_loss