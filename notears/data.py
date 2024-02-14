import numpy as np
import networkx as nx
from notears.metrics import is_dag

def generate_random_data(adjacency_matrix:np.ndarray, samples:int) -> np.ndarray:
    dim = adjacency_matrix.shape[0]
    data = np.zeros(shape=(samples, dim))

    # check if the graph is a DAG
    if not is_dag(adjacency_matrix):
        raise ValueError("The provided graph is not a directed acyclic graph (DAG).")

    # generate data based on DAG
    for var in nx.topological_sort(nx.DiGraph(adjacency_matrix)):
        # if variable has no parents -> generate data for that variable
        if np.sum(adjacency_matrix[:, var]) == 0:
            data[:, var] = np.random.uniform(
                low=-10, high=10, size=samples
            )
        # if there is an edge  -> linear relationship to parents
        else:
            data[:, var] = data @ adjacency_matrix[:, var]
        # add noise
        data[:, var] += np.random.normal(size=samples, scale=1)
    
    # normalize data to [-2, 2] interval
    data_standard = (data - np.min(data))/np.max(data)
    data = (data_standard * 4) - 2
    return data


def generate_binary_dag(n:int, alpha:float):
    """
    Generates a random Directed Acyclic Graph (DAG) represented as an adjacency matrix.

    Parameters:
    - n: Number of nodes in the graph
    - alpha: Sparseness parameter in [0, 1]

    Returns:
    - A random DAG represented as a binary adjacency matrix

    Example:
    >>> random_dag = generate_random_dag(5, 0.3)
    >>> print(random_dag)
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Parameter alpha must be in the range [0, 1]")

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Generate random edges based on the sparseness parameter alpha
    variables = np.arange(n)
    np.random.shuffle(variables)
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1 :]:
            if np.random.rand() < alpha:
                # Add a directed edge from i to j
                adjacency_matrix[var1, var2] = 1

    return adjacency_matrix


def generate_continuous_dag(n:int, alpha:float):
    """
    Generates a random Directed Acyclic Graph (DAG) represented as an adjacency matrix.

    Parameters:
    - n: Number of nodes in the graph
    - alpha: Sparseness parameter in [0, 1]

    Returns:
    - A random DAG represented as a continuous adjacency matrix

    Example:
    >>> random_dag = generate_random_dag(5, 0.3)
    >>> print(random_dag)
    """
    if not 0 <= alpha <= 1:
        raise ValueError("Parameter alpha must be in the range [0, 1]")

    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n, n), dtype=int)

    # Generate random edges based on the sparseness parameter alpha
    variables = np.arange(n)
    np.random.shuffle(variables)
    for i, var1 in enumerate(variables):
        for var2 in variables[i+1 :]:
            if np.random.rand() < alpha:
                # Add a directed edge from i to j
                adjacency_matrix[var1, var2] = np.random.choice([-1, 1]) * np.random.uniform(0.1, 2)

    return adjacency_matrix