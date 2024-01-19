import matplotlib.pyplot as plt
import numpy as np

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


def plot_confusion_matrix(conf_matrix, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(conf_matrix, cmap='Blues', interpolation='nearest')

    ax.set_colorbar()
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    classes = ['Neg', 'Pos']
    tick_marks = np.arange(len(classes))
    plt.set_xticks(tick_marks, classes)
    plt.set_yticks(tick_marks, classes)

    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center", color="black")

    plt.savefig(path)
    plt.close()


def plot_roc(fpr, tpr):
    plt.plot(fpr, tpr, label='ROC Curve')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()