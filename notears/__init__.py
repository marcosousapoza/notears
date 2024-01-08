import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from typing import Callable, Tuple


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

    # Loss calculation
    loss = (0.5/X.shape[0]) * np.linalg.norm(X - X @ W, ord='fro')**2

    # Gradient calculation. TODO: derive this!
    d_loss = -1/X.shape[0] * X.T @ (X - X @ W)

    return loss, d_loss


class NOTEARS:

    def __init__(
            self, l1:float, eps:float, c:float, omega:float,
            objective:Callable[
                [np.ndarray, np.ndarray], Tuple[float, np.ndarray]
            ] = linear_sem_loss
        ) -> None:
        self._l1 = l1
        self._W = None
        self._eps = eps
        self._c = c
        self._omega = omega
        self._objective = objective

    def fit(self, X) -> "NOTEARS":
        n,d = X.shape

        # Initial guess for W
        self._W = np.random.random(size=(d,d))
        self._a = np.random.random()

        ########################
        # Define the functions #
        ########################

        def linear_constraint(W:np.ndarray):
            # define the derivative with respect to `W`
            lc = np.trace(expm(W * W)) - d
            d_lc = expm(W * W).T * 2*W
            return lc, d_lc

        def augmented_lagrange(w:np.ndarray, rho:float, alpha:float) -> float:
            # reshape vector into a matrix
            W = w.reshape(d,d)
            # get value and gradient of objective function
            obj, d_obj = self._objective(X, W)
            d_obj = d_obj.flatten()
            # get value and proximal gradient of the l1 norm
            l1_penalty = self._l1 * np.linalg.norm(w, ord=1)
            # l1 proximal operator as gradient
            d_l1_penalty = np.sign(w) * self._l1
            # get value and gradient of the dag constraint
            lc, d_lc = linear_constraint(W)
            d_lc = d_lc.flatten()
            # compute the value and gradient of the augmented lagrange
            lag = obj + l1_penalty + 0.5*rho*lc**2 + alpha * lc
            d_lag = d_obj + d_l1_penalty + rho*lc*d_lc + alpha * d_lc
            return lag, d_lag
        
        ###############################
        # Start the fitting procedure #
        ###############################

        # get all needed parameters
        rho = 1 # initial penalty term for violations

        # define the variables to optimize over
        W_guess = self._W.copy()
        w_a = W_guess.flatten()
        alpha_a = self._a

        # calculate initial constraint violations
        lc_old, _ = linear_constraint(W_guess)
        lc_new = float('inf')

        # very large penalty can lead to overflow errors. Can be ignored i.e. just take big value
        old_settings = np.seterr(over='ignore')

        while lc_new > self._eps:
            # (a) solve primal i.e. optimize with respect to W
            while lc_new > self._c * lc_old: # check if progress rate is sufficiently fast
                res = minimize(
                    lambda w: augmented_lagrange(w, rho, alpha_a),
                    jac=True, # derivative provided
                    x0=w_a, # initial guess
                    method='L-BFGS-B'
                )
                w_b:np.ndarray = res.x
                W_guess = w_b.reshape(d,d)
                # check if linear constraint is penalized sufficiently
                lc_new, _ = linear_constraint(W_guess)
                if lc_new > self._c * lc_old:
                    rho <<= 1 # inflate by multiplying by 2

            # (b) dual ascent i.e. optimize with respect to alpha
            alpha_b:float = alpha_a + rho * lc_new

            # check termination condition i.e. linear constraint violations are minimal
            if lc_new > self._eps:
                alpha_a = alpha_b
                w_a = w_b.copy()
                lc_old = lc_new

        # reset old numpy settings
        np.seterr(**old_settings)

        # (c) return thresholded matrix or continuous matrix
        if not self._omega:
            self._W = W_guess
            return W_guess
        self._W = (W_guess > self._omega).astype(int)
        
    

