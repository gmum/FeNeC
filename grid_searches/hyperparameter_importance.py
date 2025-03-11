import argparse
import sys

import optuna
from optuna.importance import get_param_importances
from optuna.visualization import plot_param_importances

OPTUNA_DB_PATH = 'sqlite:///./results/optuna_study.db'


def analyze_hyperparameter_importance(study_name):
    """
    Analyzes hyperparameter importance for a given Optuna study.
    
    Parameters:
     - study_name (str): Name of the Optuna study in the database.
    """
    try:
        # Load the study
        study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB_PATH)

        # Compute hyperparameter importance
        param_importances = get_param_importances(study)

        # Print the importance of each hyperparameter
        print("Hyperparameter importance:")
        for param, importance in param_importances.items():
            print(f"{param}: {importance:.3f}")

        # Plot the importance (optional, comment out if not needed)
        print("\nGenerating importance plot...")
        fig = plot_param_importances(study)
        fig.show()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Analyze Optuna hyperparameter importance.")
    parser.add_argument("--study_name", type=str, required=True, help="The name of the Optuna study to analyze.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function
    analyze_hyperparameter_importance(args.study_name)
