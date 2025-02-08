import optuna
import argparse
import json
import DatasetRun
import Metrics
from KNNClassifier import KNNClassifier
from GradKNNClassifier import GradKNNClassifier
from KMeans import KMeans

def get_sampler(sampler_name):
    samplers = {
        "Random": optuna.samplers.RandomSampler(),
        "TPE": optuna.samplers.TPESampler(),
        "QMC": optuna.samplers.QMCSampler(),
        "GP": optuna.samplers.GPSampler()
    }
    return samplers.get(sampler_name, optuna.samplers.TPESampler())

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    return parser.parse_args()

def suggest_param(trial, param_name, param_config):
    """ Suggests a parameter based on fixed values, dataset-specific logic, or Optuna optimization. """
    if isinstance(param_config, (int, float, str, bool, type(None))):
        return param_config  # Fixed value

    elif isinstance(param_config, dict):
        if param_config["method"] == "suggest_float_or_none":
            return None if trial.suggest_categorical(f"{param_name}_use_null", [True, False]) else \
                   trial.suggest_float(param_name, *param_config["args"])
        
        return getattr(trial, param_config["method"])(param_name, *param_config["args"], **param_config.get("kwargs", {}))

    raise ValueError(f"Invalid parameter configuration for {param_name}")

def objective(trial, config, only_last=False):
    dataset_name = config["dataset_name"]
    study_name = config["study_name"]
    n_tasks = config["n_tasks"]
    model_type = config["model_type"]
    
    hyperparams = {param: suggest_param(trial, param, suggest) for param, suggest in config["hyperparameters"].items()}

    metric = Metrics.MahalanobisMetric(shrinkage=hyperparams["shrinkage"], gamma_1=hyperparams['gamma_1'], gamma_2=hyperparams['gamma_2'],
                                       normalization=hyperparams['metric_normalization'])
    knn_metric = Metrics.EuclideanMetric()
    kmeans = KMeans(n_clusters=hyperparams['n_clusters'], metric=knn_metric)
    
    if model_type == "KNN":
        clf = KNNClassifier(
            n_neighbors=hyperparams["n_neighbors"],
            metric=metric,
            tukey_lambda=hyperparams["tukey_lambda"],
            is_normalization=hyperparams["is_normalization"],
            kmeans=kmeans,
            device=device
        )
    else:
        clf = GradKNNClassifier(
            metric=metric,
            is_normalization=hyperparams["is_normalization"],
            tukey_lambda=hyperparams["tukey_lambda"],
            kmeans=kmeans,
            device=device,
            batch_size=hyperparams["batch_size"],
            optimizer=hyperparams["optimizer"],
            n_points=min(hyperparams["n_points"], hyperparams['n_clusters']),  # Ensures it respects min(n_points, n_clusters)
            mode=hyperparams["mode"],
            num_epochs=hyperparams["num_epochs"],
            lr=hyperparams["lr"],
            early_stop_patience=hyperparams["early_stop_patience"],
            reg_type=hyperparams["reg_type"],
            reg_lambda=hyperparams["reg_lambda"],
            normalization_type=hyperparams["normalization_type"],
            tanh_x=hyperparams["tanh_x"],
            centroids_new_old_ratio=hyperparams["centroids_new_old_ratio"],
            train_only_on_first_task=hyperparams["train_only_on_first_task"],
            dataloader_batch_size=hyperparams["dataloader_batch_size"],
            study_name=study_name,
            verbose=config["verbose"]
        )
    return DatasetRun.train(clf=clf, folder_name=f'./data/{dataset_name}', n_tasks=n_tasks, only_last=only_last, study_name=study_name, verbose=2)

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    study_name = config["study_name"]
    sampler = get_sampler(config["sampler"])
    last_accuracy_trials = config["last_accuracy_trials"]
    average_accuracy_trials = config["average_accuracy_trials"]
    verbose = config["verbose"]
    device = DatasetRun.get_device()
    
    print("Starting hyperparameter search for study:", study_name)
    DatasetRun.grid_search(objective=lambda trial: objective(trial, config,only_last=True),
                           study_name=study_name, n_trials=last_accuracy_trials, sampler=sampler,
                           restart=False, n_jobs=1, verbose=verbose)
    
    DatasetRun.grid_search(objective=lambda trial: objective(trial, config,only_last=False),
                           study_name=study_name, n_trials=average_accuracy_trials, sampler=sampler,
                           restart=False, n_jobs=1, verbose=verbose)
    
    DatasetRun.save_to_csv(study_name)
