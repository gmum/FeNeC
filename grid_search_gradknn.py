import optuna

import DatasetRun
import Metrics
from GradKNNClassifier import GradKNNClassifier
from KMeans import KMeans

device = DatasetRun.get_device()
study_name = "dataset2_gradknn"
folder_name = './data/dataset2/'

# Study parameters:
sampler = optuna.samplers.TPESampler()  # Optuna sampler
n_trials = 2  # Number of trials (different hyperparameter sets to try)

n_tasks = 6  # Number of tasks in the dataset (or number of tasks to run)
only_last = True  # Whether to predict and calculate the accuracy score only on the last task

# Constant model hyperparameters:
shrinkage = 2
metric_normalization = True
centroids_normalization = True


def objective(trial):
    # DEFINE HYPERPARAMETERS:
    n_clusters = trial.suggest_int('n_clusters', 1, 75)
    n_points = trial.suggest_int('n_points', 1, min(30, n_clusters))
    gamma_1 = trial.suggest_float('gamma_1', 0.2, 2., log=False)
    gamma_2 = trial.suggest_float('gamma_2', 0.2, 2., log=False)
    tukey_lambda = trial.suggest_float('lambda', 0.2, 1., log=False)
    when_norm = trial.suggest_categorical("when_norm", [0, 1])
    norm_type = trial.suggest_categorical("norm_type", [0, 1])
    task0_type = trial.suggest_categorical("task0_type", [0, 1])
    train_previous = trial.suggest_categorical("train_previous", [True, False])
    reg_type = trial.suggest_categorical("reg_type", [1, 2])
    lr = trial.suggest_float('lr', 0.001, 1., log=True)
    sigmoid_x = trial.suggest_float('sigmoid_x', 0.1, 10.)
    reg_lambda = trial.suggest_float('reg_lambda', .0001, 1, log=True)
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
                            train_previous=train_previous,
                            use_sigmoid=False,
                            when_norm=when_norm,
                            norm_type=norm_type,
                            task0_type=task0_type,
                            verbose=False)

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
