import optuna
import argparse

import DatasetRun
import Metrics
from GradKNNClassifier import GradKNNClassifier
from KMeans import KMeans



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






def objective(trial, dataset_name):
    
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


    # DEFINE HYPERPARAMETERS:
    n_clusters = trial.suggest_int('n_clusters', 1, 75)
    n_points = trial.suggest_int('n_points', 1, min(30, n_clusters))
    gamma_1 = trial.suggest_float('gamma_1', 0.2, 2., log=False)
    gamma_2 = trial.suggest_float('gamma_2', 0.2, 2., log=False)
    
    if(dataset_name == "dataset1"):
        tukey_lambda = 1
    elif(dataset_name == "dataset2"):
        tukey_lambda = trial.suggest_float('lambda', 0.2, 1., log=False)

    when_norm = trial.suggest_categorical("when_norm", [0, 1])
    norm_type = trial.suggest_categorical("norm_type", [0, 1])
    reg_type = trial.suggest_categorical("reg_type", [1, 2])
    lr = trial.suggest_float('lr', 0.0005, 1., log=True)
    use_sigmoid = trial.suggest_categorical("use_sigmoid", [True, False])
    sigmoid_x = trial.suggest_float('sigmoid_x', 0.1, 10.)
    reg_lambda = trial.suggest_float('reg_lambda', .0001, 1, log=True)
    add_prev_centroids = trial.suggest_categorical("add_prev_centroids", [True, False])
    only_prev_centroids = trial.suggest_categorical("only_prev_centroids", [True, False])
    repeat_prev_centroid = trial.suggest_int('repeat_prev_centroid', 1, 5)
    optimizer_type = trial.suggest_categorical('optimizer_type', ['NAdam', 'RMSprop', 'Adam'])
    dataloader_batch_size = 2 ** trial.suggest_int('dataloader_batch_size', 6, 10)
    ###

    # KNN metric:
    metric = Metrics.MahalanobisMetric(shrinkage=shrinkage, gamma_1=gamma_1, gamma_2=gamma_2,
                                       normalization=metric_normalization)
    # KMeans metric:
    knn_metric = Metrics.EuclideanMetric()

    # Initialize KMeans and KNNClassifier with defined metrics
    kmeans = KMeans(n_clusters=n_clusters, metric=knn_metric)
    clf = GradKNNClassifier(n_points=n_points,
                            mode=1,
                            metric=metric,
                            is_normalization=True,
                            tukey_lambda=tukey_lambda,
                            num_epochs=300,
                            reg_type=reg_type,
                            reg_lambda=reg_lambda,
                            lr=lr,
                            kmeans=kmeans,
                            device=device,
                            batch_size=64,
                            early_stop_patience=10,
                            train_previous=True,
                            use_sigmoid=use_sigmoid,
                            sigmoidx=sigmoid_x,
                            when_norm=when_norm,
                            norm_type=norm_type,
                            add_prev_centroids=add_prev_centroids,
                            only_prev_centroids=only_prev_centroids,
                            repeat_prev_centroid=repeat_prev_centroid,
                            optimizer_type=optimizer_type,
                            dataloader_batch_size=dataloader_batch_size,
                            verbose=False)

    # Train the classifier and return accuracy
    accuracy = DatasetRun.train(clf=clf, folder_name=folder_name, n_tasks=n_tasks,
                                only_last=only_last, verbose=False)

    return accuracy

if __name__ == "__main__":
    args = parse_args()

    
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
    DatasetRun.grid_search(objective=objective_with_args,
                        study_name=args.study_name,
                        n_trials=n_trials,
                        sampler=sampler,
                        restart=False,
                        n_jobs=1,
                        verbose=4)

    # Save the results to a CSV file
    DatasetRun.save_to_csv(args.study_name)
