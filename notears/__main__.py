from . import NOTEARS
from .helper import generate_random_dag, generate_random_data
from .helper import get_confusion_matrix
from random import randint, uniform
import numpy as np
from tqdm import tqdm
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import mlflow.pyfunc.model

NUM_EXPERIMENTS = 1000


def calculate_metrics(conf_matrix):
    TP, FP, TN, FN = conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0], conf_matrix[1, 1]

    # Calculate additional metrics
    P = TP + FN
    R = TP + FP
    FDR = (R + FP) / P if P != 0 else 0
    TPR = TP / P if P != 0 else 0
    FPR = (R + FP) / (TN + FP) if (TN + FP) != 0 else 0

    return FDR, TPR, FPR


def run_experiment():
    # get parameters for data generation
    dag_size = 20
    sparsity = uniform(0.5, 1)
    sample_size = randint(200, 2000)
    
    # generate data
    dag = generate_random_dag(dag_size, sparsity)
    data = generate_random_data(dag, sample_size)

    # get parameters for NOTEARS instance
    l1 = uniform(0, 100)
    eps = uniform(0, 1)
    c = uniform(0, 1)
    
    # fit notears
    nt = NOTEARS(l1=l1, eps=eps, c=c, omega=False)
    dag_found = nt.fit(data)
    
    # post process the results
    for omega in np.linspace(0, 1, 50):
        # threshold
        dag_binary = (dag_found > omega).astype(int)
        # calculate the findings
        conf_matrix = get_confusion_matrix(dag, dag_binary)
        FDR, TPR, FPR = calculate_metrics(conf_matrix)

        # Log results with MLflow
        with mlflow.start_run():
            mlflow.log_param("dag_size", dag_size)
            mlflow.log_param("sparsity", sparsity)
            mlflow.log_param("sample_size", sample_size)
            mlflow.log_param("l1", l1)
            mlflow.log_param("eps", eps)
            mlflow.log_param("c", c)
            mlflow.log_param("omega", omega)
            
            # Log metrics
            mlflow.log_metric("fdr", FDR)
            mlflow.log_metric("tpr", TPR)
            mlflow.log_metric("fpr", FPR)
            mlflow.log_metric("shd", np.sum(np.abs(dag - dag_binary)))
            
            
if __name__ == '__main__':
    # Run tests on sparse big graphs
    mlflow.set_tracking_uri('sqlite:///experiment.db')
    for _ in tqdm(range(NUM_EXPERIMENTS)):
        run_experiment()
