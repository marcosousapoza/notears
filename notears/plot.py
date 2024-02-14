import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from matplotlib.figure import Figure
from notears.metrics import find_dag_violation_threshold
import numpy as np


def plot_graph_from_adjacency_matrix(adj_matrix:np.ndarray) -> Figure:
    """
    Plots a graph based on the given adjacency matrix using NetworkX and matplotlib.
    
    Parameters:
    - adj_matrix (np.ndarray): A square numpy array representing the adjacency matrix of the graph.
    """
    # Create a graph from the adjacency matrix
    G = nx.DiGraph(adj_matrix)
    
    # Define node labels in LaTeX style
    labels = {i: r'$X_{{{}}}$'.format(i + 1) for i in range(len(G.nodes))}
    
    # Draw the graph
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(
        G, pos, labels=labels, with_labels=True, 
        node_size=700, node_color='skyblue', font_size=16, 
        font_weight='bold', arrows=True, ax=ax
    )
    return fig

def number_edges_plot(W:np.ndarray) -> Figure:
    W_abs = np.abs(W)
    flat = np.sort(W_abs.flatten())[::-1] 
    n_edges = np.arange(1, len(flat) + 1)
    threshold = find_dag_violation_threshold(
        W_abs
    )
    # Plot results
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(flat, n_edges)
    ax.set_xscale('log')
    ax.set_ylabel('# edges')
    ax.set_xlabel('threshold')
    ax.set_title('Edges given Threshold')
    ax.axvline(x=threshold, color='r', linestyle='--', label='DAG Violation Threshold')
    ax.legend()
    fig.tight_layout()
    return fig

def plot_roc_curve(tpr:np.ndarray, fpr:np.ndarray, thresholds:np.ndarray) -> Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    # Plot tpr and fpr
    ax.plot(fpr, tpr)
    # Target FPR values for annotations
    target_tprs = np.array([0, 0,1, 0.2, 0,3, 0.4, 0.5, 0.6, 0.8, 1.0])
    # Find the indices where the FPR is closest to the target FPRs
    idx = np.searchsorted(tpr, target_tprs, side="right")
    # Ensure indices are within the bounds of the fpr array
    idx = np.minimum(idx, len(tpr) - 1)
    # Plot and annotate the selected thresholds
    ax.scatter(fpr[idx], tpr[idx], marker='+')
    for i in idx:
        ax.text(fpr[i], tpr[i], f'{thresholds[i]:.2f}', fontsize=9)  # Annotating the curve
    # Plot y=x line i.e. "random setting edges"
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    fig.tight_layout()
    return fig


def difference_plot(true:np.ndarray, pred:np.ndarray, binary:np.ndarray) -> Figure:
    # Create images
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    # Plot the continuous predicted dag
    cmap = sns.diverging_palette(240, 10, s=80, l=55, as_cmap=True)
    abs_max = np.max(np.abs(pred))
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    ax[0].imshow(pred, cmap=cmap, norm=norm)
    ax[0].set_title('Found Continuous')
    # Plot the binary predicted graph
    ax[1].imshow(binary, cmap='gray')
    ax[1].set_title('Found Binary')
    # Plot the true DAG
    ax[2].imshow(true, cmap='gray')
    ax[2].set_title('True DAG')
    # Plot the differences of the ture and binary dag
    diff_pixels = true != binary
    ax[3].imshow(diff_pixels, cmap='gray')
    ax[3].set_title('Differences')
    fig.tight_layout()
    return fig


def plot_confusion_matrix(conf_matrix, cmap='Blues'):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    classes = ['E=1', 'E=0']
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap=cmap, 
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_ylabel('Found graph')
    ax.set_xlabel('True graph')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    return fig
