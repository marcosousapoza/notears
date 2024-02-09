from random import uniform, randint
from notears import NOTEARS
from notears.test import run_experiment
from notears.data import (
    generate_binary_dag, 
    generate_continuous_dag,
    generate_random_data,
)
from notears.metrics import (
    calculate_fdr, calculate_fpr, calculate_shd, calculate_tpr,
    confusion_matrix, roc_auc, find_dag_violation_threshold
)
from notears.plot import (
    difference_plot, plot_confusion_matrix, 
    plot_roc_curve, number_edges_plot
)
from notears.loss import linear_sem_loss
import numpy as np
from datetime import datetime

def experiment():
    # get parameters for this experiment
    l1 = uniform(0.001, 10)
    eps = 1e-7
    c = uniform(0.05, 0.8)
    dag_size = randint(5, 10)
    sparsity = uniform(0.1, 1)
    sample_size = randint(20, 200)
    params = {
        'l1':l1,
        'eps':eps,
        'c':c,
        'dag_size':dag_size,
        'sparsity':sparsity,
        'sample_size':sample_size
    }
    # generate data
    W_true = generate_continuous_dag(dag_size, sparsity)
    W_true_binary = (W_true > 0).astype(int)
    data = generate_random_data(W_true, sample_size)
    # fit notears and get results
    nt = NOTEARS(l1=l1, eps=eps, c=c, objective=linear_sem_loss)
    start = datetime.now()
    nt.fit(data)
    duration = (datetime.now() - start).total_seconds()
    W_pred = nt.get_discovered_graph()
    threshold = find_dag_violation_threshold(W_pred)
    omega = uniform(threshold, np.max(W_pred)) # sample feasible omega
    params['omega'] = omega
    W_pred_binary = nt.get_discovered_graph(omega)
    # calculate metrics
    conf_matrix = confusion_matrix(W_true_binary, W_pred_binary)
    FDR = calculate_fdr(W_true_binary, W_pred_binary)
    TPR = calculate_tpr(W_true_binary, W_pred_binary)
    FPR = calculate_fpr(W_true_binary, W_pred_binary)
    SHD = calculate_shd(W_true_binary, W_pred_binary)
    loss = linear_sem_loss(data, W_pred)[0]
    tprs, fprs, thresholds, AUC = roc_auc(W_true_binary, W_true, num_thresholds=1000)
    metrics = {"loss":loss, "fdr":FDR,"tpr":TPR,"fpr":FPR,"shd":SHD,"auc":AUC,"duration":duration}
    # create figures
    figures = [
        (plot_roc_curve(tprs, fprs, thresholds), 'roc'),
        (plot_confusion_matrix(conf_matrix), 'confusion_matrix'),
        (difference_plot(W_true_binary, W_pred, W_pred_binary), 'difference_plot'),
        (number_edges_plot(W_pred), 'edgecount')
    ]
    return params, metrics, figures

if __name__ == '__main__':
    db = 'sqlite:///experiment.db'
    name = 'medium2'
    run_experiment(db, name, 30, experiment)