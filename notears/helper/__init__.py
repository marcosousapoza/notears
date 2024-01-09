import numpy as np
import networkx as nx
from itertools import product
from typing import Generator
import matplotlib.pyplot as plt

def plot_graphs(true_dag, found_dag, path):
    # Create images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    # Plot the true DAG
    ax[0].imshow(true_dag, cmap='gray')
    ax[0].set_title('True DAG')
    # Plot the best DAG with differences highlighted
    ax[1].imshow(found_dag, cmap='gray')
    ax[1].set_title('Best DAG')
    diff_pixels = true_dag != found_dag
    ax[2].imshow(diff_pixels, cmap='gray')
    ax[2].set_title('Differences')
    # Save the figure
    plt.savefig(path)
    plt.close()

def generate_random_dag(n:int, alpha:float):
    """
    Generates a random Directed Acyclic Graph (DAG) represented as an adjacency matrix.

    Parameters:
    - n: Number of nodes in the graph
    - alpha: Sparseness parameter in [0, 1]

    Returns:
    - A random DAG represented as an adjacency matrix

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


def check_dag(adjacency_matrix:np.ndarray):
    # create dag instance
    dag = nx.DiGraph(adjacency_matrix)
    # check if the graph is a DAG
    return nx.is_directed_acyclic_graph(dag)


def generate_random_data(adjacency_matrix:np.ndarray, samples:int) -> np.ndarray:
    dim = adjacency_matrix.shape[0]
    data = np.zeros(shape=(samples, dim))

    # check if the graph is a DAG
    if not check_dag(adjacency_matrix):
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
        data[:, var] += np.random.normal(size=samples, scale=3)
    
    return data


def generate_all_dags(dim: int) -> Generator[np.ndarray, None, None]:
    """
    Generate all possible directed acyclic graphs (DAGs) as binary matrices.

    Parameters:
    - dim (int): Dimension of the square matrix.

    Yields:
    - numpy.ndarray: Binary matrix representing a directed acyclic graph.
    """
    assert dim > 0, 'dimension must be greater than 0'

    dags = product([0, 1], repeat=dim*dim)
    for graph in dags:
        graph_np = np.array(graph).reshape(dim, dim)
        if check_dag(graph_np):
            yield graph_np


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
