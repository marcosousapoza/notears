import networkx as nx
import numpy as np


def check_dag(adjacency_matrix:np.ndarray):
    # create dag instance
    dag = nx.DiGraph(adjacency_matrix)
    # check if the graph is a DAG
    return nx.is_directed_acyclic_graph(dag)