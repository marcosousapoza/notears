import matplotlib.pyplot as plt

def plot_graphs_(true_dag, found_dag, path):
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