import sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from typing import Callable, Tuple, TypedDict, List
from notears.loss import LossFunction

class MetaData(TypedDict):
    iterations:int
    rho:int
    lagrangian_value:List[float]
    converged:bool

def weight_init(X: np.ndarray, ees:float, ses:float) -> np.ndarray:
    """Generates initial guess for W_0

    Args:
        X (np.ndarray): Data matrix
        ees (float): Expected effect size
        ses (float): Expected standard deviation of effect size

    Returns:
        np.ndarray: Initialized weight matrix
    """
    n, d = X.shape
    # Normalize the arrays to create probability distributions
    W = np.random.uniform(0, 1, size=(d,d))
    np.fill_diagonal(W, 0)
    return W


class NOTEARS:

    def __init__(
            self, l1: float, eps: float, c: float,
            objective: LossFunction
    ) -> None:
        self._l1 = l1
        self._W = None
        self._eps = eps
        self._c = c
        self._objective = objective
        self._reset_meta_data()

    def _reset_meta_data(self):
        self._meta:MetaData = {
            'iterations':0,
            'rho':0,
            'lagrangian_value':[],
            'dag_constraint_value':[],
            'converged':False
        }

    def set_params(self, **params):
        params_adapted = {'_'+key:value for key, value in params.items()}
        for param, value in params_adapted.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise AttributeError(f"{param} is not a valid parameter.")
        return self

    def get_discovered_graph(self, omega:float=None) -> np.ndarray:
        if not omega:
            return self._W
        return (np.abs(self._W) > omega).astype(int)
    
    def get_meta_data(self) -> MetaData:
        return self._meta.copy()

    def fit(self, X, W_init:np.ndarray=None, alpha_init:float=None) -> "NOTEARS":
        n, d = X.shape

        # Initial guess for W
        self._meta.clear()
        self._W = W_init if isinstance(W_init, np.ndarray) else weight_init(X, 1, 0.25)
        self._a = alpha_init if alpha_init != None else np.random.random()
        # Reset meta data collection
        self._reset_meta_data()

        ########################
        # Define the functions #
        ########################

        def linear_constraint(W:np.ndarray):
            # define the derivative with respect to `W`
            lc = expm(W * W).trace() - d
            d_lc = expm(W * W).T * 2*W
            return lc, d_lc

        def augmented_lagrange(w:np.ndarray, rho:float, alpha:float) -> Tuple[float, float]:
            # reshape vector into a matrix
            W = w.reshape(d,d)
            # get value and gradient of objective function
            obj, d_obj = self._objective.loss(X, W)
            d_obj = d_obj.reshape(-1)
            # get value and proximal gradient of the l1 norm
            l1_penalty = self._l1 * np.linalg.norm(w, ord=1)
            # l1 proximal operator as gradient
            d_l1_penalty = np.sign(w) * self._l1
            # get value and gradient of the dag constraint
            lc, d_lc = linear_constraint(W)
            d_lc = d_lc.reshape(-1)
            # compute the value and gradient of the augmented lagrange
            lag = obj + l1_penalty + 0.5*rho*lc**2 + alpha * lc
            d_lag = d_obj + d_l1_penalty + rho*lc*d_lc + alpha * d_lc
            return lag, d_lag
        
        ###############################
        # Start the fitting procedure #
        ###############################

        # get all needed parameters
        rho = 1.0 # initial penalty term for violations

        # define the variables to optimize over
        W_a:np.ndarray = self._W.copy()
        w_a = W_a.flatten()
        alpha_a = self._a
        rho_max = sys.float_info.max # maximum value for rho -> avoid overflow

        # calculate initial constraint violations
        lc_old, _ = linear_constraint(W_a)
        lc_new = float('inf')

        # very large penalty can lead to overflow errors. Can be ignored i.e. just take big value
        old_settings = np.seterr(over='ignore', invalid='ignore')
        # specify bounds for the optimisation function
        bounds = [(0, 0) if i == j else (0, None) 
                for i in range(d) for j in range(d)]
        
        while lc_new > self._eps and rho < rho_max:
            self._meta['iterations'] += 1
            # (a) solve primal i.e. optimize with respect to W
            while lc_new > self._c * lc_old and rho < rho_max: # check if progress rate is sufficiently fast
                res = minimize(
                    lambda w: augmented_lagrange(w, rho, alpha_a),
                    jac=True, # derivative provided
                    x0=w_a, # initial guess
                    method='L-BFGS-B',
                    bounds=bounds
                )
                w_b:np.ndarray = res.x
                # log result
                lg, _ = augmented_lagrange(w_b, rho, alpha_a)
                self._meta['lagrangian_value'].append(lg)
                # reshape result into matrix
                W_guess = w_b.reshape(d,d)
                lc_new, _ = linear_constraint(W_guess)
                # check if linear constraint is penalized sufficiently
                if lc_new > self._c * lc_old:
                    rho = 10*rho # inflate by multiplying by itself

            # (b) dual ascent i.e. optimize with respect to alpha
            alpha_b:float = alpha_a + rho * lc_new

            # check termination condition i.e. linear constraint violations are minimal
            if lc_new > self._eps:
                alpha_a = alpha_b
                w_a = w_b.copy()
                lc_old = lc_new

        # reset old numpy settings
        np.seterr(**old_settings)
        # update meta data
        self._meta['rho'] = rho
        self._meta['converged'] = lc_new < self._eps

        # (c) return thresholded matrix or continuous matrix
        self._W = W_guess.copy()
        return self