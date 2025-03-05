import abc

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score

from kmeans import KMeans
from metrics import MahalanobisMetric


class Classifier(abc.ABC):
    def __init__(self, metric, data_normalization=False, tukey_lambda=1., kmeans=None, device='cpu', batch_size=8,
                 config_arguments=None, *args, **kwargs):
        """
        Initializes the Classifier.

        Parameters:
         - metric (Metric): The distance metric to be used.
         - data_normalization (bool): Indicates whether data normalization should be applied.
         - tukey_lambda (float): Lambda value for Tukey’s Ladder of Powers transformation.
         - kmeans (KMeans): Optional KMeans object for clustering, if used.
         - device (str): Device on which computations are performed ('cpu' or 'cuda').
         - batch_size (int): Batch size for splitting test data. Used in prediction.
         - config_arguments (dict): Additional configuration parameters from the inherited class for wandb config.
        """
        self.metric = metric
        self.data_normalization = data_normalization
        self.tukey_lambda = tukey_lambda
        self.kmeans = kmeans
        self.device = device
        self.batch_size = batch_size
        self.D_centroids = torch.tensor([], device=device)

        # Configure wandb with local and keyword arguments of valid types.
        self.config = {key: value for key, value in {**locals(), **kwargs, **config_arguments}.items() if
                       isinstance(value, (str, int, float, bool))}
        if isinstance(self.kmeans, KMeans):
            self.config.update(self.kmeans.get_config())
        if isinstance(self.metric, MahalanobisMetric):
            self.config.update(self.metric.get_config())

    def fit(self, D, **kwargs):
        """
        Fits the model to the training data. It can be called multiple times. On each call, the new data
        (new classes) is processed and concatenated with the old one.

        Parameters:
         - D (torch.Tensor): Training data tensor of shape [n_classes, samples_per_class, n_features].
                             It can also be a list of tensors if the number of samples per class isn't equal.
        """
        if torch.is_tensor(D):
            if D.ndim == 2:
                D = D.unsqueeze(0)  # Ensure data has the correct shape.
            D = D.type(torch.float32).to(self.device)
        else:
            D = [d.type(torch.float32).to(self.device) for d in D]

        # Process the data:
        # self.D stores only the current task's data, with Tukey transformation applied.
        #  It will be used for preprocessing the metric and possibly in children classes
        #  shape: [n_classes (only in this task), samples_per_class, n_features]
        # self.D_centroids stores the centroids across all tasks, with Tukey transformation
        #  and optional normalization applied. It will be used during prediction.
        #  shape: [n_classes (across all tasks), n_centroids, n_features]

        if torch.is_tensor(D):
            D_centroids = D.clone()  # Clone the data, so that we can perform clustering without Tukey applied
        else:
            D_centroids = [d.clone() for d in D]

        # Apply Tukey transformation to current task data and preprocess for the distance metric.
        self.D = self.apply_tukey(D)
        self.metric.preprocess(self.D)

        if self.kmeans is not None:
            if isinstance(self.kmeans, KMeans):  # If it's the custom implementation of KMeans:
                self.kmeans.metric_preprocess(self.D)  # Preprocess data for KMeans metric (used for Mahalanobis).
                if torch.is_tensor(D):
                    D_centroids = self.kmeans.fit_predict(D_centroids)  # Perform KMeans clustering.
                else:
                    D_centroids = torch.cat([self.kmeans.fit_predict(d.unsqueeze(0)) for d in D])
            else:  # Otherwise, if using sklearn's implementation:
                # Perform KMeans clustering for each class separately and stack the results into a tensor.
                D_centroids = torch.stack([torch.tensor(self.kmeans.fit(d_class.cpu().numpy()).cluster_centers_)
                                          .to(self.device) for d_class in D_centroids])
                if self.tukey_lambda != 1:
                    D_centroids = torch.clip(D_centroids, min=0)

        D_centroids = self.apply_tukey(D_centroids)  # Apply Tukey transformation to centroids.
        if self.data_normalization:
            # Normalize centroids if normalization is enabled.
            D_centroids = self.normalize_data(D_centroids)
            self.D = self.normalize_data(self.D)  # Normalize the data if normalization is enabled.

        self.D_centroids = torch.concat((self.D_centroids, D_centroids))
        self.n_classes = len(self.D_centroids)

        return self

    @abc.abstractmethod
    def model_predict(self, distances):
        """
        Abstract method for predicting class labels based on distances to training samples.

        Parameters:
         - distances (Tensor): Distances of shape [batch_size, n_classes, n_centroids].

        Returns:
         - Tensor: Predicted class labels.
        """
        pass

    def predict(self, X):
        """
        Predicts the class labels for the input data X.

        Parameters:
         - X (torch.Tensor): Test data tensor of shape [n_samples, n_features].

        Returns:
         - torch.Tensor: Predicted class labels for the test data.
        """
        # Process the test data
        X = self.apply_tukey(X)
        if self.data_normalization:
            X = self.normalize_data(X)

        return self.metric.calculate_batch(self.model_predict, self.D_centroids, X, self.batch_size)

    def get_config(self):
        """
        Returns:
         - config (dict): Dictionary containing local variables and keyword arguments.
        """
        return self.config

    def apply_tukey(self, T):
        """ Applies Tukey’s Ladder of Powers transformation to the tensor T. """
        if torch.is_tensor(T):
            if self.tukey_lambda != 0:
                return torch.pow(T, self.tukey_lambda)
            else:
                return torch.log(T)
        else:
            if self.tukey_lambda != 0:
                return [torch.pow(t, self.tukey_lambda) for t in T]
            else:
                return [torch.log(t) for t in T]

    @staticmethod
    def normalize_data(T, epsilon=1e-8):
        """ Normalizes the data tensor T """
        if torch.is_tensor(T):
            if T.ndim == 3:
                # Normalize class-based data (3D tensor)
                T_permuted = T.permute(0, 2, 1)
                norm = torch.linalg.norm(T_permuted, dim=1, ord=2, keepdim=True)
                representation = T_permuted / (norm + epsilon)
                return representation.permute(0, 2, 1)
            else:
                # Normalize sample-based data (2D tensor)
                T_permuted = T.T
                norm = torch.linalg.norm(T_permuted, dim=0, ord=2, keepdim=True)
                representation = T_permuted / (norm + epsilon)
                return representation.T
        else:
            return [Classifier.normalize_data(t) for t in T]

    @staticmethod
    def getD(X, y):
        """
        Transforms the input data X and labels y into a tensor D for training.

        Parameters:
         - X (torch.Tensor): Data tensor of shape [n_samples, n_features].
         - y (torch.Tensor): Labels tensor of shape [n_samples].

        Returns:
         - torch.Tensor: Transformed tensor D of shape [n_classes, samples_per_class, n_features].
        """
        n_classes = len(torch.unique(y))  # Determine number of unique classes.
        D = [[] for _ in range(n_classes)]  # Create a list for each class.
        for _X, _y in zip(X, y):
            D[_y].append(_X)  # Group samples by their class.

        # Ensure each class has the same number of samples by trimming to minimum size.
        min_len = min([len(d) for d in D])
        D = [d[:min_len] for d in D]  # Uniform number of samples per class.
        return torch.stack([torch.stack(D[i]) for i in range(n_classes)]).type(torch.float32).to(X.device)

    @staticmethod
    def accuracy_score(y_true, pred, verbose=False, task_sizes=None):
        """ Calculates the accuracy score. """
        if verbose and task_sizes is not None:
            precision, recall, fscore, support = score(y_true.detach().cpu().numpy(), pred.detach().cpu().numpy(),
                                                       zero_division=0.0)

            conf_matrix = confusion_matrix(y_true.detach().cpu().numpy(), pred.detach().cpu().numpy())
            accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)

            prev_task_size = 0
            precision_tasks, recall_tasks, fscore_tasks, p_answers_tasks, accuracy_tasks = [], [], [], [], []
            for task in task_sizes:
                precision_tasks.append(precision[prev_task_size:prev_task_size + task].mean())
                recall_tasks.append(recall[prev_task_size:prev_task_size + task].mean())
                fscore_tasks.append(fscore[prev_task_size:prev_task_size + task].mean())
                p_answers_tasks.append(((prev_task_size <= pred) & (pred < prev_task_size + task)).sum() / len(y_true))
                accuracy_tasks.append(accuracy[prev_task_size:prev_task_size + task].mean())
                prev_task_size += task

            # Create DataFrame for formatted output
            df = pd.DataFrame({"Task": list(range(len(task_sizes))),
                               "Class num": task_sizes,
                               "Precision": [f"{p * 100:.2f}%" for p in precision_tasks],
                               "Recall": [f"{r * 100:.2f}%" for r in recall_tasks],
                               "FScore": [f"{f:.2f}" for f in fscore_tasks],
                               "Accuracy": [f"{a:.2f}" for a in accuracy_tasks],
                               "% of all Answers": [f"{p * 100:.2f}%" for p in p_answers_tasks]})

            print(df.to_markdown(index=False))

        return torch.sum(torch.eq(y_true, pred)).item() / len(y_true) * 100
