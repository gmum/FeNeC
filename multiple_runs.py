import optuna
import torch
import argparse
import json
import DatasetRun
import Metrics
from KNNClassifier import KNNClassifier
from GradKNNClassifier import GradKNNClassifier
from KMeans import KMeans
import statistics

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization with Optuna.")
    parser.add_argument("--config", type=str, required=True, help="Path to the JSON config file.")
    return parser.parse_args()

def get_params(hyperparams, run_id):
    params = {}
    for key, value in hyperparams.items():
        if isinstance(value, list):
            params[key] = value[run_id]
        else:
            params[key] = value
    return params


def run_on_params(config, data_path, run_ind, only_last=False):
    study_name = config["study_name"]
    n_tasks = config["n_tasks"]
    model_type = config["model_type"]

    hyperparams = config["hyperparameters"]
    hyperparams = get_params(hyperparams, run_ind)
    

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
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        clf = GradKNNClassifier(
            metric=metric,
            is_normalization=hyperparams["is_normalization"],
            tukey_lambda=hyperparams["tukey_lambda"],
            kmeans=kmeans,
            device="cuda" if torch.cuda.is_available() else "cpu",
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
    return DatasetRun.train(clf=clf, folder_name=data_path, n_tasks=n_tasks, only_last=only_last, study_name=study_name, return_all_accuracies=True, verbose=config["verbose"])

if __name__ == "__main__":
    args = parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    for dataset_name, dataset_info in config.items():
        print("-" * 50)
        print("-" * 50)
        print("-" * 50)
        print(f"Dataset: {dataset_name}")
        print(f"Number of tasks: {dataset_info['n_tasks']}")
        print(f"Model type: {dataset_info['model_type']}")


        for i in range(dataset_info["n_runs"]):
            print(f"Run {i + 1}")
            accuracies = []
            avg_accuracies = []
            accuracies_averaged_by_seed = [0 for _ in range(dataset_info["n_tasks"])]
            avg_accuracies_averaged_by_seed = 0
            std_last_accuracy = 0
            std_avg_accuracy = 0

            print("-" * 50)
            print("-" * 50)


            for data_path in dataset_info["data_paths"]:
                print("-" * 50)
                print(f"Data path: {data_path}")
                hyperparams = dataset_info["hyperparameters"]
                params = get_params(hyperparams, i)
                print(f"Hyperparameters: {params}")

                accuracies.append(run_on_params(dataset_info, data_path, i, only_last=False))





            for i in range(len(accuracies)):
                avg_accuracies.append(sum(accuracies[i]) / len(accuracies[i]))
            for i in range(len(dataset_info["data_paths"])):
                for j in range(dataset_info["n_tasks"]):
                    accuracies_averaged_by_seed[j] += accuracies[i][j]
            
            for i in range(dataset_info["n_tasks"]):
                accuracies_averaged_by_seed[i] /= len(dataset_info["data_paths"])

            avg_accuracies_averaged_by_seed = sum(accuracies_averaged_by_seed) / len(accuracies_averaged_by_seed)

            last_accuracies = [a[-1] for a in accuracies]
            std_last_accuracy = statistics.stdev(last_accuracies)

            std_avg_accuracy = statistics.stdev(avg_accuracies)


            print("Accuracies:")
            for accuracy in accuracies:
                print(accuracy)
            print(f"Average accuracies: {avg_accuracies}")
            print(f"Accuracies averaged by seed: {accuracies_averaged_by_seed}")
            print(f"Average accuracies averaged by seed: {avg_accuracies_averaged_by_seed}")
            print(f"Standard deviation of last accuracy: {std_last_accuracy}")
            print(f"Standard deviation of average accuracy: {std_avg_accuracy}")
