"""
A file containing functions used for testing, running models, and performing grid search using Optuna.
"""

import math
import re
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import wandb

# Path to the Optuna database for storing study results
OPTUNA_DB_PATH = 'sqlite:///./results/optuna_study.db'


def train(clf, folder_name, n_tasks, only_last=False, study_name=None, return_all_accuracies=False, verbose=0):
    """
    Trains a classifier on a series of tasks, optionally testing on the test set after each task.

    Parameters:
     - clf (Classifier.Classifier): The classifier model to train.
     - folder_name (str): Path to the folder containing the HDF5 task files.
     - n_tasks (int): Number of tasks to train on.
     - only_last (bool): If True, evaluate on the test set only for the last task. If False, evaluate on all tasks.
     - study_name (str): Name of the Optuna study for logging(only used if verbose >= 2).
     - return_all_accuracies (bool): If True, return the accuracy on all tasks.
                                     Do not set this to True while grid searching, only during experiments.
     - verbose (int): Verbosity level for logging.

    Returns:
     - float: The accuracy on the final task.
    """
    # Initialize W&B logging if verbose is high enough
    if verbose >= 2:
        if study_name is None:
            raise ValueError("study_name must be provided for logging")

        with open("wandb_key.txt", "r") as key_file:
            api_key = key_file.read().strip()

        # Login to W&B using the key and get config from clf
        wandb.login(key=api_key)
        wandb.init(project=study_name,
                   config=clf.get_config(),
                   settings=wandb.Settings(init_timeout=300))

    device = clf.device
    task_sizes = []
    accuracies = []
    accuracy_sum = 0
    start_time = time.time()

    # Loop over the tasks to train and test the classifier.
    for task_number in range(n_tasks):
        current_file = f"{folder_name}/task_{task_number}.hdf5"

        with h5py.File(current_file, "r") as f:
            # Load training and testing data from the HDF5 file
            X_train = torch.tensor(f["X_train"][:], dtype=torch.float32, device=device)
            y_train = torch.tensor(f["y_train"][:], dtype=torch.float32, device=device)

            X_test = torch.tensor(f["X_test"][:], dtype=torch.float32, device=device)
            y_test = torch.tensor(f["y_test"][:], dtype=torch.float32, device=device)

            # Group training samples by class
            D = [X_train[y_train == y_class] for y_class in y_train.unique()]

            # Check whether all the classes have the same number of samples
            equal_samples_num = [d.size(0) for d in D].count(D[0].size(0)) == len(D)
            if equal_samples_num:  # If so, convert them into a tensor
                D = torch.concat([d.unsqueeze(0) for d in D])
            task_sizes.append(len(D))

            # Determine whether to train the classifier (if required) and later use it for prediction
            should_predict = (not only_last or task_number == n_tasks - 1)

            if should_predict and verbose:
                start = time.time()  # Track the time for performance analysis.

            # Fit the classifier to the grouped data
            clf.fit(D, task_num=task_number, train=should_predict, study_name=study_name, verbose=verbose)

            # If prediction is enabled, generate predictions and calculate accuracy on the test set
            if should_predict:
                pred = clf.predict(X_test)
                accuracy = clf.accuracy_score(y_test, pred, verbose=verbose, task_sizes=task_sizes)
                accuracy_sum += accuracy
                accuracies.append(accuracy)
                if verbose >= 1:
                    end = time.time()
                    print(f'task {task_number}: (time: {(end - start):.4f}s)')
                    print(f"FeCAM accuracy: {f['info'].attrs['accuracy']:.4f}; My accuracy: {accuracy:.4f}")
                if verbose >= 2:
                    wandb.log({"task": task_number, f"task_{task_number}_accuracy": accuracy})

    # Finish the W&B run if verbose is high enough
    if verbose >= 2:
        if not only_last:
            wandb.log({f"average_accuracy": accuracy_sum / n_tasks})
        wandb.finish()

    if verbose >= 1:
        print("Total time: ", time.time() - start_time)

    if return_all_accuracies:
        return accuracies

    return accuracy


def get_device(verbose=True):
    """ Returns the device to be used ('cuda' if GPU is available, otherwise 'cpu'). """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print(f'Device used: {device}')
    return device


def is_jupyter():
    """ Checks if the code is being run inside a Jupyter notebook. """
    return 'ipykernel' in sys.modules


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


def save_to_csv(study_name, path='./results/', only_complete=True):
    """
    Saves the results of the Optuna study to a CSV file.

    Parameters:
     - study_name (str): Name of the Optuna study.
     - path (str): Path to the results folder
     - only_complete (bool): If True, only include completed trials.
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


def format_param_name(param_name):
    """
    Converts parameter names like 'gamma_1' into LaTeX-style symbols ('$\\gamma_{1}$').

    Parameters:
     - param_name (str): The parameter name to format (e.g., "gamma_1", "lambda_2", "alpha", "n_points").
    """
    # A Map of parameters that needs to be changed in the final
    param_name_map = {"tukey_lambda": "lambda", "lr": "Learning rate",
                      "n_points": "$N_{points}$", "n_clusters": "$N_{clusters}$", "n_neighbors": "$N_{neighbors}$"}
    if param_name in param_name_map:
        param_name = param_name_map[param_name]

    # List of Greek letters that should be replaced with LaTeX symbols
    greek_letters = ["alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda", "sigma", "omega"]

    # Regular expression to extract:
    # - A base name (e.g., "gamma", "lambda")
    # - Any additional text after the main name (e.g., "knn" in "gamma_1 knn")
    match = re.match(r"([a-zA-Z]+)(?:_(\d+))?(.*)", param_name)

    if match:
        base, subscript, extra_text = match.groups()  # Extract base name, optional subscript, and extra text

        # If the base name is a recognized Greek letter, format it as LaTeX
        if base in greek_letters:
            formatted = fr"$\{base}$"  # Converts "gamma" → "$\gamma$"
            if subscript:
                formatted = formatted[:-1] + f"_{{{subscript}}}$"  # Adds subscript: "$\gamma$" → "$\gamma_{1}$"
            return formatted + extra_text  # Preserve any extra text (e.g., " knn")

    # If the name is not in the Greek letter list, return it unchanged
    return param_name


def format_param_accuracy_title(param_name):
    formatted_param_name = format_param_name(param_name)
    if formatted_param_name[0] == '$':
        return r'$\text{Accuracy vs }' + formatted_param_name[1:]
    else:
        return r'$\text{Accuracy vs ' + formatted_param_name + '}$'


def plot_param_accuracy(param_name, param_vals, accuracies, ylim=True, ylim_set=None, xlim_set=None,
                        only_later=None, fig_size=(10, 6), font_scale=1.5, path_to_pdf=None, title=None,
                        title_pad=None, label_pad=None, ax=None):
    """
    Create a Seaborn scatter plot of accuracy vs. a given parameter.

    Parameters:
     - param_name (str): The name of the parameter (e.g., "gamma_1").
     - param_vals (np.ndarray): An array of parameter values (x-axis values).
     - accuracies (np.ndarray): An array of accuracies corresponding to each run (y-axis values).
     - ylim (bool): If True, remove points with unusually low accuracy (using the rule: below mean - std).
     - ylim_set (float): If not None, set it as the y-axis lower limit.
     - ylim_set (tuple of float): If not None, set it as the x-axis limits.
     - only_later (float): If not None, use only the later part of the data (between 0 and 1).
     - fig_size (tuple of int): Figure size in inches, e.g. (width, height). Default is (10, 6).
     - font_scale (float): Scaling factor for fonts (labels, ticks). Increase for a paper‐quality figure.
     - path_to_pdf (str): If not None, the plot will be saved to the given file path in PDF format.
     - title (str): Title of the plot. If None, no title is added.
     - title_pad (int): Padding around the title to make the plots consistent.
     - label_pad (int): Padding around the x label to make the plots consistent.
     - ax: If provided, the plot is drawn on this axis (for subplot support).
    """
    # Remove first data samples: For example for Optuna grid searches which are less representative then
    if only_later:
        new_starting_pos = int((1 - only_later) * len(param_vals))
        param_vals = param_vals[new_starting_pos:]
        accuracies = accuracies[new_starting_pos:]

    # Apply y-axis filtering: Remove low-accuracy outliers
    if ylim:
        lower_bound = accuracies.mean() - accuracies.std() if ylim_set is None else ylim_set
        keep = accuracies >= lower_bound
        param_vals = param_vals[keep]
        accuracies = accuracies[keep]

    # Apply x-axis limit:
    if xlim_set is not None:
        keep = (xlim_set[0] <= param_vals) & (param_vals <= xlim_set[1])
        param_vals = param_vals[keep]
        accuracies = accuracies[keep]

    # Create a Pandas DataFrame for Seaborn
    df = pd.DataFrame({param_name: param_vals, "Accuracy": accuracies})

    # Set up Seaborn's styling for a clean, paper-quality plot
    sns.set_context("paper", font_scale=font_scale)
    sns.set_style("whitegrid", {"grid.linestyle": "--"})

    # If no axis (ax) is provided, create a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)

    # Create the scatter plot
    sns.scatterplot(x=param_name, y="Accuracy", data=df, s=50, alpha=0.6, ax=ax)

    # Format the parameter name and title for display (e.g., convert "gamma_1" → "$\gamma_{1}$")
    formatted_param_name = format_param_name(param_name)
    ax.set_xlabel(formatted_param_name, fontsize=font_scale * 12, labelpad=label_pad)
    ax.set_ylabel(r"$\text{Accuracy (%)}$", fontsize=font_scale * 12)

    # Add a title if provided
    if title:
        ax.set_title(title, fontsize=font_scale * 14, fontweight="bold", pad=title_pad)

    # Adjust tick label size for readability
    ax.tick_params(axis="both", which="major", labelsize=font_scale * 10)

    # Improve layout to prevent overlapping elements
    plt.tight_layout()

    # Save to PDF if a file path is provided
    if path_to_pdf is not None:
        plt.savefig(path_to_pdf, format="pdf", dpi=300, bbox_inches='tight', transparent=True, pad_inches=0.01)

    if ax is None:
        plt.show()  # Display the plot


def plot_params_accuracy(study_name, path='./results/', columns=3, ylim=True, only_later=None):
    """
    Plots multiple hyperparameter versus accuracy graphs from a CSV file with Optuna Study.

    Parameters:
     - study_name (str): Name of the study.
     - path (str): Path to the results' folder.
     - columns (int): Number of columns for the plot grid.
     - ylim (bool): Whether to set y-axis limits based on accuracy distribution.
     - only_later (float): If not None, use only the later part of the data (between 0 and 1).
    """
    # Load data
    df = load_from_csv(study_name, path)
    accuracies = df['value'].values

    # Extract hyperparameter names
    params = [key for key in df.keys() if key.startswith('params_')]

    # Determine grid size
    rows = math.ceil(len(params) / columns)

    # Create a shared figure and axes grid
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 7.5, rows * 5.5))
    axes = np.array(axes).reshape(-1)  # Flatten to make indexing easier

    # Loop through parameters and plot
    for i, param in enumerate(params):
        if np.isscalar(df[param].values[0]):  # Ensure it's a numeric parameter
            plot_param_accuracy(param_name=param[7:],  # Remove "params_" prefix
                                param_vals=df[param].values,
                                accuracies=accuracies,
                                ylim=ylim,
                                only_later=only_later,
                                title=format_param_accuracy_title(param[7:]),
                                ax=axes[i])  # Pass the corresponding subplot axis

    # Remove empty subplots (if any)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
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


def plot_gradknn_parameters(classes, file_path='data.csv', n_cols=3, row_height=3, col_width=5):
    data = pd.read_csv(file_path)

    # Prepare the grid layout
    n_rows = (len(classes) + n_cols - 1) // n_cols  # Calculate number of rows needed
    fig_size = (n_cols * col_width, n_rows * row_height)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size)
    axes = axes.flatten()  # Flatten axes for easier indexing

    for idx, class_x in enumerate(classes):
        # Generate column names for the class
        r_col = f'r_{class_x}'
        alpha_col = f'alpha_{class_x}'
        b_col = f'b_{class_x}'
        a_col = f'a_{class_x}'

        # Check if the columns exist
        if not all(col in data.columns for col in [r_col, alpha_col, b_col, a_col]):
            print(f"Columns for class {class_x} do not exist in the dataset.")
            continue

        # Calculate cumulative sum of epochs for the x-axis
        cumulative_epochs = range(len(data))

        # Extract necessary data
        r_values = data[r_col]
        alpha_values = data[alpha_col]
        b_values = data[b_col]
        a_values = data[a_col]

        # Plot in the corresponding subplot
        ax = axes[idx]
        ax.plot(cumulative_epochs, r_values, label=f'{r_col}', color='blue')
        ax.plot(cumulative_epochs, alpha_values, label=f'{alpha_col}', color='orange')
        ax.plot(cumulative_epochs, b_values, label=f'{b_col}', color='green')
        ax.plot(cumulative_epochs, a_values, label=f'{a_col}', color='red')

        # Add vertical lines for epoch resets
        epoch_zeros = data.index[data['epoch'] == 0].tolist()
        for epoch_zero in epoch_zeros:
            ax.axvline(x=epoch_zero, color='gray', linestyle='--', alpha=0.7)

        # Customize subplot
        ax.set_title(f'Class {class_x}')
        ax.set_ylim([-5, 5])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)

    # Hide any unused subplots
    for idx in range(len(classes), len(axes)):
        axes[idx].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def analyze_hyperparameter_importance(study_name):
    """
    Analyzes hyperparameter importance for a given Optuna study.

    Parameters:
     - study_name (str): Name of the Optuna study in the database.
    """
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_PATH)

    # Compute hyperparameter importance
    param_importance = optuna.importance.get_param_importances(study)

    # Print the importance of each hyperparameter
    print("Hyperparameter importance:")
    for param, importance in param_importance.items():
        print(f"{param}: {importance:.3f}")

    # Plot the importance (optional, comment out if not needed)
    print("\nGenerating importance plot...")
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()
