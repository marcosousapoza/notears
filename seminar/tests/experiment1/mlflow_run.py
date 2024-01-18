from seminar.notears import NOTEARS
from seminar.helper.data import generate_binary_dag, generate_random_data
from seminar.evaluation.metric import get_confusion_matrix

from random import randint, uniform
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.pyfunc.model

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

NUM_EXPERIMENTS = 10
EXPERIMENT_NAME = 'neg-weights-5n'

def calculate_metrics(conf_matrix):
    TP, FP, TN, FN = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]

    # Calculate additional metrics
    P = TP + FN
    R = TP + FP
    FDR = (R + FP) / P if P != 0 else 0
    TPR = TP / P if P != 0 else 0
    FPR = (R + FP) / (TN + FP) if (TN + FP) != 0 else 0

    return FDR, TPR, FPR


def run_experiment(ex_id):
    # get parameters for data generation
    dag_size = 5
    sparsity = uniform(0.2, 1)
    sample_size = randint(100, 200)
    
    # generate data
    dag = generate_binary_dag(dag_size, sparsity)
    data = generate_random_data(dag, sample_size)

    # get parameters for NOTEARS instance
    l1 = uniform(0.001, 10)
    eps = 1e-6
    c = uniform(0.8, 1)
    
    # fit notears
    nt = NOTEARS(l1=l1, eps=eps, c=c, omega=False)
    dag_found = nt.fit(data)

    # get best omega based on shanon distance
    omega_best = np.inf
    shd_best = np.inf
    best_dag = None
    for omega in np.linspace(0, 1, 50):
        dag_binary = (dag_found > omega).astype(int)
        shd = np.sum(np.abs(dag - dag_binary))
        if shd <= shd_best:
            omega_best = omega
            shd_best = shd
            best_dag = dag_binary.copy()


    with mlflow.start_run(experiment_id=ex_id) as run:

        # make image and log
        path = f'./img/{run.info.run_id}.png'
        plot_graphs(dag, best_dag, path)
        mlflow.log_artifact(path)

        # log results with MLflow
        mlflow.log_param("dag_size", dag_size)
        mlflow.log_param("sparsity", sparsity)
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("l1", l1)
        mlflow.log_param("eps", eps)
        mlflow.log_param("c", c)
        mlflow.log_param("omega", omega_best)

        # calculate the findings
        conf_matrix = get_confusion_matrix(dag, dag_binary)
        FDR, TPR, FPR = calculate_metrics(conf_matrix)
    
        # log metrics
        mlflow.log_metric("fdr", FDR)
        mlflow.log_metric("tpr", TPR)
        mlflow.log_metric("fpr", FPR)
        mlflow.log_metric("shd", np.sum(np.abs(dag - dag_binary)))
        
            
if __name__ == '__main__':
    # Run tests on sparse big graphs
    mlflow.set_tracking_uri('sqlite:///experiment.db')
    ex = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if ex != None:
        ex_id = ex._experiment_id
    else:
        ex_id = mlflow.create_experiment(EXPERIMENT_NAME)
    for _ in tqdm(range(NUM_EXPERIMENTS)):
        run_experiment(ex_id)
