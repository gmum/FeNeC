# FeNeC & FeNeC-Log: Continual Learning Classifiers

## Overview

This repository contains the implementation of **FeNeC (Feature Neighborhood Classifier)** and **FeNeC-Log**, as introduced in the paper [FeNeC: Enhancing Continual Learning via Feature Clustering with Neighbor- or Logit-based Classification](https://arxiv.org/pdf/2503.14301).

These methods address Class-Incremental Learning scenarios by leveraging feature-based representations, data clustering and distance metrics. **FeNeC** employs a nearest neighbor approach, while **FeNeC-Log** extends it with trainable parameters, enabling more flexible and adaptive classification.

## Usage

#### Install dependencies

Run the following command to install the required libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

#### Prepare your dataset

Place your dataset (feature representations from a model such as ResNet or ViT) into the `data/` folder. Detailed guidelines on dataset structure and format are provided in the [Data and Project Structure](#data-and-project-structure) section.

#### Explore example notebooks

Refer to the [Jupyter notebooks](#notebooks) for example usage and experiments:

- `cifar_vit.ipynb`: Includes all evaluated methods on CIFAR datasets with ViT representations.
- `cifar_resnet.ipynb`: Focuses on FeNeC and FeNeC-Log with ResNet representations.

These notebooks walk through the usage and evaluation of our methods.


# Data and Project Structure

## Folder and File Organization

```
├── data/                    # Place your dataset HDF5 files here
│   ├── dataset_name/
│   │   ├── task_0.hdf5
│   │   ├── task_1.hdf5
│   │   └── ...
│
├── FeNeC.py                 # FeNeC implementation
├── FeNeC_Log.py             # FeNeC-Log implementation
├── MLPClassifier.py         # Obsolete classifier
├── KMeans.py                # Custom KMeans clustering implementation       
├── Metrics.py               # Distance metrics (Euclidean, Cosine, Mahalanobis)
├── utils.py                 # Helper functions (training, evaluation, visualization tools, etc.)
├── tests.py                 # Unit tests for core components
├── grid_searches/           # Optuna grid searches and experiments 
│   ├── *.json               # Search spaces and experiment configs
│   └── *.py                 # Scripts used to run Optuna experiments on a server
│
└── notebooks/               # Example Jupyter notebooks
    ├── cifar_vit.ipynb      # All methods evaluated
    ├── cifar_resnet.ipynb   # FeNeC and FeNeC-Log
    ├── 2d_embeddings.ipynb  # Representations embedded with t-SNE and UMAP
    ├── kmeans.ipynb         # Custom KMeans test
    ├── sample_mnist.ipynb   # FeNeC with Mahalanobis tests
    └── sample_moons.ipynb   # Similar tests

```

## Dataset Format (`data/dataset_name/`)

Each task representation should be stored in separate `.hdf5` file, for example: `task_0.hdf5`, `task_1.hdf5`, etc.

Each file must include:

-   `X_train`: training features (shape `[num_samples, num_features]`)
-   `y_train`: training labels (shape `[num_samples]`)
-   `X_test`: test features (shape `[num_samples, num_features]`)
-   `y_test`: test labels (shape `[num_samples]`)

***Note:** These files are not provided in this repository.*


# Classifier Implementations

## Base Class: Classifier

An abstract class providing shared functionality for all classifiers.

### Hyperparameters

-   **metric**: Distance metric for classification (`Euclidean`, `Cosine`, `Mahalanobis`).
-   **data_normalization**: Whether to normalize the data (important if the input features aren't already normalized).
-   **tukey_lambda**: Lambda parameter for Tukey’s Ladder of Powers transformation. Controls the strength of the transformation and can be seen as a regularization force.
-   **kmeans**: Optional custom KMeans object for clustering-based sample selection.
-   **device**: Computation device (`'cpu'` or `'cuda'`).
-   **batch_size**: Controls batch size for distance computations. Impacts speed and memory usage but does not affect outputs.


## FeNeC (Feature Neighborhood Classifier)

A nearest-neighbor classifier leveraging distance metrics for classification. No additional training is required beyond storing and accessing task data.

### Hyperparameters

-   **n_neighbors**: Number of neighbors considered when classifying new samples.

## FeNeC-Log (Feature Logit Classifier)

An extension of FeNeC that introduces trainable parameters and makes decisions based on calculated logits.

### Hyperparameters

-   **optimizer**: Optimization algorithm (e.g., `'SGD'`).
-   **n_points**: Number of representative points retained per class for future tasks.
-   **num_epochs**: Number of training epochs.
-   **lr**: Learning rate for the optimizer.
-   **early_stop_patience**: Number of epochs to wait before early stopping if the validation loss does not improve.
-   **train_only_on_first_task**: If `True`, training is only performed on the first task with no further updates in future tasks.
-   **mode**:
    -   `0`: Shared parameters across all classes (recommended and used in the paper).
    -   `1`: Separate parameters for each class (obsolete, generally worse results).

Additional hyperparameters are available for `mode=1` (refer to the class documentation). Mode `0` is recommended.


# Custom KMeans Implementation

A custom KMeans clustering algorithm, implemented in `kmeans.py`, a key component of both FeNeC and FeNeC-Log.

### Hyperparameters

-   **n_clusters**: Number of clusters to form.
-   **max_iter**: Maximum number of iterations.
-   **tol**: Tolerance for the stopping criterion; stops if centroids move less than this threshold.
-   **metric**: Distance metric used (`Euclidean`, `Cosine`, `Mahalanobis`).
-   **seed**: Random seed for reproducibility.

For examples and comparisons with scikit-learn's KMeans, refer to `kmeans.ipynb`.


# Distance Metrics

Implemented in `metrics.py`, used in both FeNeC and FeNeC-Log.

-   **EuclideanMetric**: Standard L2 distance.
-   **CosineMetric**: `1 - cosine similarity`.
-   **MahalanobisMetric**: Mahalanobis distance, requiring covariance matrix estimation. 

**Hyperparameters for Mahalanobis**:
-   **shrinkage**: Shrinkage type (`0`: none, `1`: normal, `2`: double).
-   **gamma_1**: Diagonal shrinkage factor.
-   **gamma_2**: Off-diagonal shrinkage factor.
-   **normalization**: Whether to normalize the covariance matrix.


# Grid Search & Experiments

All hyperparameter search definitions and experiment configurations are located in the `grid_searches/` folder.

-   `*.json`: Search spaces and experiment configurations for FeNeC and FeNeC-Log on various datasets.
-   `grid_search.py`: Runs grid searches using Optuna.
-   `hyperparameter_importance.py`: Analyzes hyperparameter importance from Optuna studies.
-   `multiple_runs.py`: Executes multiple training runs for statistical analysis.


# Utilities & Tests

-   `utils.py`: Helper functions for training, evaluation, grid search, and visualization tools (e.g., plots, performance metrics).
-   `tests.py`: Unit tests for core components to ensure correctness and stability.


# Notebooks

For experimentation, visualization, and debugging:

-   `cifar_resnet.ipynb`: Experiments on CIFAR datasets with ResNet representations (FeNeC and FeNeC-Log).
-   `cifar_vit.ipynb`: Experiments on CIFAR datasets with ViT representations (includes obsolete methods).
-   `2d_embeddings.ipynb`: t-SNE and UMAP embeddings for visualization.
-   `kmeans.ipynb`: Comparisons between the custom KMeans and scikit-learn’s implementation.
-   `sample_mnist.ipynb`: Testing Mahalanobis distance and FeNeC on MNIST.
-   `sample_moons.ipynb`: Experiments on synthetic datasets.

***Note:** If import issues arise in the notebooks, try moving them to the main directory.*

# Code Authors

**Hubert Jastrzębski**: Designed and implemented the core logic, including FeNeC and FeNeC-Log classifiers, distance metrics, and custom KMeans. Developed Jupyter notebooks, visualization tools, and unit tests. 

**Krzysztof Pniaczek**: Led large-scale experiment execution, performance analysis, and hyperparameter optimization using Optuna. Integrated wandb for experiment tracking and visualization.

# Cite

If you use our code, please cite the following paper:

    @misc{ksiazek2025fenec,
          title={FeNeC: Enhancing Continual Learning via Feature Clustering with Neighbor- or Logit-Based Classification}, 
          author={Kamil Książek and Hubert Jastrzębski and Bartosz Trojan and Krzysztof Pniaczek and Michał Karp and Jacek Tabor},
          year={2025},
          eprint={2503.14301},
          archivePrefix={arXiv},
          primaryClass={cs.LG},
          url={https://arxiv.org/abs/2503.14301}, 
    }
