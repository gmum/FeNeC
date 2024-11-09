import optuna

import DatasetRun
import Metrics
from KMeans import KMeans
from KNNClassifier import KNNClassifier

device = DatasetRun.get_device()
study_name = "dataset1"
folder_name = '../data/dataset1'

# Study parameters:
sampler = optuna.samplers.RandomSampler()  # Optuna sampler
n_trials = 3  # Number of trials (different hyperparameter sets to try)

n_tasks = 10  # Number of tasks in the dataset (or number of tasks to run)
only_last = True  # Whether to predict and calculate the accuracy score only on the last task

# Constant hyperparameters:
shrinkage = 1
tukey_lambda = 1
metric_normalization = True
centroids_normalization = False


def objective(trial):
    # DEFINE HYPERPARAMETERS:
    n_clusters = trial.suggest_int('n_clusters', 1, 250)
    n_neighbors = trial.suggest_int('n_neighbors', 1, min(50, n_clusters))
    gamma_1 = trial.suggest_float('gamma_1', 0.001, 6., log=True)
    gamma_2 = trial.suggest_float('gamma_2', 0.001, 6., log=True)
    ###

    # KNN metric:
    metric = Metrics.MahalanobisMetric(shrinkage=shrinkage, gamma_1=gamma_1, gamma_2=gamma_2,
                                       normalization=metric_normalization)
    # KMeans metric:
    knn_metric_type = trial.suggest_categorical("knn_metric", ["euclidean", "mahalanobis"])
    if knn_metric_type == "euclidean":
        knn_metric = Metrics.EuclideanMetric()
    elif knn_metric_type == "mahalanobis":
        knn_metric = Metrics.MahalanobisMetric(shrinkage=shrinkage, gamma_1=gamma_1, gamma_2=gamma_2,
                                               normalization=metric_normalization)

    # Initialize KMeans and KNNClassifier with defined metrics
    kmeans = KMeans(n_clusters=n_clusters, metric=knn_metric)
    clf = KNNClassifier(n_neighbors=n_neighbors, metric=metric, is_normalization=centroids_normalization,
                        tukey_lambda=tukey_lambda, kmeans=kmeans, device=device)

    # Train the classifier and return accuracy
    accuracy = DatasetRun.train(clf=clf, folder_name=folder_name, n_tasks=n_tasks,
                                only_last=only_last, verbose=False)

    return accuracy


# Start the study
print("Starting the hyperparameter search for study:", study_name)
DatasetRun.grid_search(objective=objective,
                       study_name=study_name,
                       n_trials=n_trials,
                       sampler=sampler,
                       restart=False,
                       n_jobs=1,
                       verbose=4)

# Save the results to a CSV file
DatasetRun.save_to_csv(study_name)
