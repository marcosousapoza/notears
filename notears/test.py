from typing import Callable, Tuple, Dict, Any, List
from datetime import datetime
from matplotlib.figure import Figure
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow


def run_experiment(
        db:str, name:str, n:int,
        experiment:Callable[[None], Tuple[
            Dict[str, Any],
            Dict[str, float],
            List[Tuple[Figure, str]]
        ]]
    ):
    
    mlflow.set_tracking_uri(db)
    #mlflow.set_tracking_uri('sqlite:///experiment.db')
    ex = mlflow.get_experiment_by_name(name)
    if ex != None:
        ex_id = ex._experiment_id
    else:
        ex_id = mlflow.create_experiment(name)
    for _ in tqdm(range(n)):
        # run custom experiment
        params, metrics, figures = experiment()
        with mlflow.start_run(experiment_id=ex_id):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for fig, name in figures:
                path = f"{name}.png"
                mlflow.log_figure(fig, path)
            plt.close('all')