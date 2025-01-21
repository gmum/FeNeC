import optuna
import argparse

import DatasetRun
import Metrics
from KNNClassifier import KNNClassifier
from KMeans import KMeans

dataset_name = ""
study_name = ""

def get_sampler(sampler_name):
    """Get the Optuna sampler based on its name."""
    if sampler_name == "Random":
        return optuna.samplers.RandomSampler()
    elif sampler_name == "Grid":
        return optuna.samplers.GridSampler()
    elif sampler_name == "TPE":
        return optuna.samplers.TPESampler()
    elif sampler_name == "CMAES":
        return optuna.samplers.CmaEsSampler()
    elif sampler_name == "NSGAII":
        return optuna.samplers.NSGAIISampler()
    elif sampler_name == "QMC":
        return optuna.samplers.QMCSampler()
    elif sampler_name == "GP":
        return optuna.samplers.GPSampler()
    elif sampler_name == "BoTorch":
        return optuna.samplers.BoTorchSampler()
    else:
        raise ValueError(f"Unsupported sampler: {sampler_name}")

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna.")
    parser.add_argument("--n_trials", type=int, default=1, help="Number of trials for hyperparameter search.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder.")
    parser.add_argument("--sampler", type=str, default="TPE", choices=[
        "Random", "Grid", "TPE", "CMAES", "NSGAII", "QMC", "GP", "BoTorch"], help="Sampler to use for Optuna.")
    parser.add_argument("--study_name", type=str, required=True, help="Name of the Optuna study.")
    parser.add_argument("--only_last",type=bool, default=True, help="Whether to predict and calculate the accuracy score only on the last task.")
    return parser.parse_args()






def objective(trial):
    shrinkage = 0
    metric_normalization = True
    centroids_normalization = True
    n_tasks = 0
    if(dataset_name == "dataset1"):
        n_tasks = 10
        shrinkage = 1
        metric_normalization = True
        centroids_normalization = False
    elif(dataset_name == "dataset2"):
        n_tasks = 6
        shrinkage = 2
        metric_normalization = True
        centroids_normalization = True
    elif(dataset_name == "dataset3"):
        n_tasks = 6
        shrinkage = 2
        metric_normalization = True
        centroids_normalization = True

    
      # DEFINE HYPERPARAMETERS:
    n_clusters = trial.suggest_int('n_clusters', 1, 100)
    n_neighbors = trial.suggest_int('n_neighbors', 1, min(50, n_clusters))
    gamma_1 = trial.suggest_float('gamma_1', 0.001, 6., log=True)
    gamma_2 = trial.suggest_float('gamma_2', 0.001, 6., log=True)
    tukey_lambda = trial.suggest_float('lambda', 0.001, 3., log=True)
    ###

    # KNN metric:
    metric = Metrics.MahalanobisMetric(shrinkage=shrinkage, gamma_1=gamma_1, gamma_2=gamma_2,
                                       normalization=metric_normalization)
    # KMeans metric:
    knn_metric = Metrics.EuclideanMetric()

    # Initialize KMeans and KNNClassifier with defined metrics
    kmeans = KMeans(n_clusters=n_clusters, metric=knn_metric)
    clf = KNNClassifier(n_neighbors=n_neighbors, metric=metric, is_normalization=centroids_normalization,
                        tukey_lambda=tukey_lambda, kmeans=kmeans, device=device)
    

    accuracy = DatasetRun.train(clf=clf, folder_name=folder_name, n_tasks=n_tasks,
                                only_last=only_last,study_name = study_name,verbose=2)
    return accuracy

if __name__ == "__main__":
    args = parse_args()

    dataset_name = args.dataset
    study_name = args.study_name
    
    device = DatasetRun.get_device()
    folder_name = f'./data/{args.dataset}'

    # Study parameters:
    sampler = get_sampler(args.sampler)  # Optuna sampler to use
    n_trials = args.n_trials  # Number of trials (different hyperparameter sets to try)
    only_last = args.only_last  # Whether to predict and calculate the accuracy score only on the last task

    # Constant model hyperparameters:



    def objective_with_args(trial):
        return objective(trial, args.dataset)



    # Start the study
    print("Starting the hyperparameter search for study:", args.study_name)
    DatasetRun.grid_search(objective=objective,
                        study_name=args.study_name,
                        n_trials=n_trials,
                        sampler=sampler,
                        restart=False,
                        n_jobs=1,
                        verbose=4)

    # Save the results to a CSV file
    DatasetRun.save_to_csv(args.study_name)
