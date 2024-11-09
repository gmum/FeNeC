"""
A file containing functions used for testing, running models, and performing grid search using Optuna.
"""

import math
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch

# Path to the Optuna database for storing study results
OPTUNA_DB_PATH = 'sqlite:///./results/optuna_study.db'


def get_device(verbose=True):
    """ Returns the device to be used ('cuda' if GPU is available, otherwise 'cpu'). """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Device used: {device}')
    return device


def is_jupyter():
    """ Checks if the code is being run inside a Jupyter notebook. """
    return 'ipykernel' in sys.modules


def train(clf, folder_name, n_tasks, only_last=False, verbose=False):
    """
    Trains a classifier on a series of tasks, optionally testing on the test set after each task.

    Parameters:
     - clf (Classifier.Classifier): The classifier model to train.
     - folder_name (str): Path to the folder containing the HDF5 task files.
     - n_tasks (int): Number of tasks to train on.
     - only_last (bool): If True, evaluate on the test set only for the last task. If False, evaluate on all tasks.
     - verbose (bool): If True, print task details and timing information.

    Returns:
     - float: The accuracy on the final task.
    """
    device = clf.device

    # Loop over the tasks to train and test the classifier.
    for task_number in range(n_tasks):
        current_file = f"{folder_name}/task_{task_number}.hdf5"

        with h5py.File(current_file, "r") as f:
            # Load training and testing data from the HDF5 file
            X_train = torch.tensor(np.array(f["X_train"]), dtype=torch.float32, device=device)
            y_train = torch.tensor(np.array(f["y_train"]), dtype=torch.float32, device=device)

            X_test = torch.tensor(np.array(f["X_test"]), dtype=torch.float32, device=device)
            y_test = torch.tensor(np.array(f["y_test"]), dtype=torch.float32, device=device)

            # Group training samples by class
            D = torch.concat([X_train[y_train == y_class].unsqueeze(0) for y_class in y_train.unique()])

            if verbose:
                start = time.time()  # Track the time for performance analysis.

            # Determine whether to train the classifier (if required) and later use it for prediction
            should_predict = (not only_last or task_number == n_tasks - 1)

            # Fit the classifier to the grouped data
            clf.fit(D, train=should_predict)

            # If prediction is enabled, generate predictions and calculate accuracy on the test set
            if should_predict:
                pred = clf.predict(X_test)
                accuracy = clf.accuracy_score(y_test, pred)

            if verbose:
                end = time.time()
                print(f'task {task_number}: (time: {(end - start):.4f}s)')
                print(f"FeCAM accuracy: {f['info'].attrs['accuracy']:.4f}; My accuracy: {accuracy:.4f}")

    return accuracy


def grid_search(objective, study_name, n_trials, sampler=optuna.samplers.TPESampler(), restart=False, n_jobs=1,
                verbose=3):
    """
    Performs a grid search over hyperparameters using Optuna.

    Parameters:
     - objective (function): Objective function for optimization.
     - study_name (str): Name of the Optuna study.
     - n_trials (int): Number of trials to run.
     - sampler (object): Optuna sampler for hyperparameter space exploration (default is TPE).
     - restart (bool): Whether to delete the previous study.
     - n_jobs (int): Number of parallel jobs to run during the search.
     - verbose (int): Verbosity level for logging.
    """
    # Set verbosity levels based on user input
    verbose_levels = [optuna.logging.CRITICAL, optuna.logging.ERROR, optuna.logging.WARNING,
                      optuna.logging.INFO, optuna.logging.DEBUG]
    optuna.logging.set_verbosity(verbose_levels[verbose])

    # Optionally restart the study
    if restart:
        optuna.delete_study(study_name=study_name, storage=OPTUNA_DB_PATH)

    # Create or load an Optuna study
    study = optuna.create_study(sampler=sampler,
                                direction='maximize',
                                study_name=study_name,
                                storage=OPTUNA_DB_PATH,
                                load_if_exists=True)

    # Perform the hyperparameter optimization
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Print the best results if verbosity is high enough
    if verbose >= 3:
        print("Best hyperparameters: ", study.best_params)
        print("Best accuracy: ", study.best_value)


def save_to_csv(study_name, path='./results/', only_complete=True):
    """
    Saves the results of the Optuna study to a CSV file.

    Parameters:
     study_name (str): Name of the Optuna study.
     path (str): Path to the results folder
     only_complete (bool): If True, only include completed trials.
    """
    # Load Optuna study to Pandas DataFrame
    loaded_study = optuna.load_study(
        study_name=study_name,
        storage=OPTUNA_DB_PATH
    )

    df = loaded_study.trials_dataframe()

    # Remove incomplete trials if specified
    if only_complete:
        df.drop(df[df.state != 'COMPLETE'].index, inplace=True)

    # Save to CSV
    df.to_csv(f"{path}{study_name}.csv", index=False)


def load_from_csv(study_name, path='./results/'):
    """ Loads the results of an Optuna study from a CSV file. """
    return pd.read_csv(f"{path}{study_name}.csv")


def plot_accuracy_trials(study_name, path='./results/', ylim=True):
    """
    Plots accuracy over trials from a CSV file with Optuna Study.

    Parameters:
     - study_name (str): Name of the Optuna study.
     - path (str): Path to the results' folder.
     - ylim (bool): Whether to set the y-axis limits based on data spread.
    """
    # Load data from csv to Pandas DataFrame
    df = load_from_csv(study_name, path)
    accuracies = df['value'].values

    # Plot the accuracies across trials
    plt.plot(accuracies)

    # If specified, ignore accuracies to far from the mean
    if ylim:
        plt.ylim(bottom=(accuracies.mean() - accuracies.std()))

    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over trials')
    plt.show()


def plot_hyperparameter(param_name, param_vals, accuracies, deg=2, ylim=True):
    """
    Plots a scatter plot of hyperparameter values against accuracies and fits a polynomial.

    Parameters:
     - param_name (str): Name of the hyperparameter.
     - param_vals (list or array): Values of the hyperparameter.
     - accuracies (list or array): Corresponding accuracy values.
     - deg (int): Degree of the polynomial fit.s
     - ylim (bool): Whether to limit the y-axis based on data.
    """
    # If specified, ignore accuracies to far from the mean
    if ylim:
        mask = accuracies.mean() - accuracies.std() <= accuracies
        param_vals = param_vals[mask]
        accuracies = accuracies[mask]

    # Scatter plot of parameter values and accuracies
    plt.scatter(param_vals, accuracies, color='blue', s=5, alpha=0.6)

    # Fit a polynomial of specified degree to the data
    z = np.polyfit(param_vals, accuracies, deg)
    p = np.poly1d(z)
    x_range = np.linspace(min(param_vals), max(param_vals), 500)
    plt.plot(x_range, p(x_range), "g-", linewidth=2)

    plt.xlabel(param_name)
    plt.ylabel('accuracy')
    plt.title(f'{param_name} vs accuracy')
    plt.grid(True)


def plot_hyperparameters(study_name, path='./results/', columns=3, deg=2, ylim=True):
    """
    Plots multiple hyperparameter versus accuracy graphs from a CSV file with Optuna Study.

    Parameters:
     - study_name (str): Name of the study.
     - path (str): Path to the results' folder.
     - columns (int): Number of columns for the plot grid.
     - deg (int): Degree of polynomial fit for each hyperparameter plot.
     - ylim (bool): Whether to set y-axis limits based on accuracy distribution.
    """
    df = load_from_csv(study_name, path)
    accuracies = df['value'].values

    params = []
    for key in df.keys():
        if key.startswith('params_'):
            params.append(key)

    rows = math.ceil(len(params) / columns)
    width, height = 7.5, 5.5
    plt.figure(figsize=(columns * width, rows * height))

    for i, param in enumerate(params):
        if np.isreal(df[param].values[0]):
            plt.subplot(rows, columns, i + 1)
            plot_hyperparameter(param[7:], df[param].values, accuracies, deg, ylim)

    plt.show()


def print_results(study_name, path='./results/', only_important=True):
    """
    Prints the sorted results of an Optuna study.

    Parameters:
     - study_name (str): Name of the study.
     - path (str): Path to the results' folder.
     - only_important (bool): If True, only prints the accuracy and hyperparameter values.

    Returns:
     - pd.DataFrame: Sorted DataFrame with the top results.
    """
    df = load_from_csv(study_name, path)
    df_sorted = df.sort_values(by=['value'], ascending=False)

    # Optionally drop unimportant columns
    if only_important:
        for key in df_sorted.keys():
            if key.startswith('params_'):
                df_sorted[key[7:]] = df_sorted[key]
            if key != "value":
                df_sorted.drop(key, axis=1, inplace=True)

    return df_sorted
