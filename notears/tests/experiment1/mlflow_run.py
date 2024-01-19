from notears.notears_algorithm import NOTEARS
from notears.helper.data import generate_binary_dag, generate_random_data
from notears.tests.plot import plot_graphs, plot_confusion_matrix
from notears.evaluation.metrics import (
    confusion_matrix, 
    calculate_fdr, calculate_shd,
    calculate_fpr, calculate_tpr,
    roc_auc
)
from random import randint, uniform
import os
from tqdm import tqdm
import mlflow

NUM_EXPERIMENTS = 10
EXPERIMENT_NAME = 'neg-weights-5n'


def run_experiment(ex_id):
    # get parameters for this experiment
    l1 = uniform(0.001, 10)
    eps = 1e-6
    c = uniform(0.8, 1)
    omega = uniform(0, 0.1)
    dag_size = 5
    sparsity = uniform(0.2, 1)
    sample_size = randint(100, 200)
    # generate data
    W_true = generate_binary_dag(dag_size, sparsity)
    W_true_binary = (W_true > 0).astype(int)
    data = generate_random_data(W_true, sample_size)
    # fit notears and get results
    nt = NOTEARS(l1=l1, eps=eps, c=c, omega=False)
    nt.fit(data)
    W_pred = nt.get_discovered_graph()
    W_pred_binary = nt.get_discovered_graph(omega)
    # calculate metrics
    conf_matrix = confusion_matrix(W_true_binary, W_pred_binary)
    FDR = calculate_fdr(W_true_binary, W_pred_binary)
    TPR = calculate_tpr(W_true_binary, W_pred_binary)
    FPR = calculate_fpr(W_true_binary, W_pred_binary)
    SHD = calculate_shd(W_true_binary, W_pred_binary)
    fprs, tprs, AUC = roc_auc(W_true_binary, W_true, num_thresholds=1000)

    with mlflow.start_run(experiment_id=ex_id) as run:
        # make image and log
        path = f'./img/{run.info.run_id}.png'
        plot_graphs(W_true_binary, W_pred_binary, path)
        mlflow.log_artifact(path)

        # log results with MLflow
        mlflow.log_params(
            {"dag_size":dag_size, "sparsity":sparsity,
             "sample_size":sample_size, "l1":l1,
             "eps":eps, "c":c, "omega":omega}
        )
        # log metrics
        mlflow.log_metrics(
            {"fdr":FDR,"tpr":TPR,"fpr":FPR,"shd":SHD,"auc":AUC}
        )

        
            
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
