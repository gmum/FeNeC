import abc

import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support as score


class Classifier(abc.ABC):
    def __init__(self, metric, is_normalization=False, tukey_lambda=1., kmeans=None, device='cpu', batch_size=8):
        """
        Initializes the Classifier.

        Parameters:
         - metric (Metric): The distance metric to be used.
         - is_normalization (bool): Indicates whether data normalization should be applied.
         - tukey_lambda (float): Lambda value for Tukey’s Ladder of Powers transformation.
         - kmeans (KMeans): Optional KMeans object for clustering, if used.
         - device (str): Device on which computations are performed ('cpu' or 'cuda').
         - batch_size (int): Batch size for splitting test data. Used in prediction.
        """
        self.metric = metric
        self.is_normalization = is_normalization
        self.tukey_lambda = tukey_lambda
        self.kmeans = kmeans
        self.device = device
        self.batch_size = batch_size
        self.is_first_fit = True

    def apply_tukey(self, T):
        """ Applies Tukey’s Ladder of Powers transformation to the tensor T. """
        if self.tukey_lambda != 0:
            return torch.pow(T, self.tukey_lambda)
        else:
            return torch.log(T)

    def fit(self, D, **kwargs):
        """
        Fits the model to the training data.
        It can be called multiple times (is_first_fit=True only on the first call).
        On each call, the new data (new classes) is processed and concatenated with the old one.

        Parameters:
         - D (torch.Tensor): Training data tensor of shape [n_classes, samples_per_class, n_features].
                             In subsequent calls, the samples_per_class and n_features dimensions
                             must match the initial call.
        """
        if D.ndim == 2:
            D = D.unsqueeze(0)  # Ensure data has the correct shape.
        D = D.type(torch.float32).to(self.device)

        # Process the data:
        # self.D stores only the current task's data, with Tukey transformation applied.
        #  It will be used for preprocessing the metric and possibly in children classes
        #  shape: [n_classes (only in this task), samples_per_class, n_features]
        # self.D_centroids stores the centroids across all tasks, with Tukey transformation
        #  and optional normalization applied. It will be used during prediction.
        #  shape: [n_classes (across all tasks), n_centroids, n_features]

        D_centroids = D.clone()  # Clone the data, so that we can perform clustering without Tukey applied

        self.D = self.apply_tukey(D)  # Apply Tukey transformation to current task data.
        self.metric.preprocess(self.D)  # Preprocess for the distance metric.

        if self.kmeans is not None:
            self.kmeans.metric_preprocess(self.D)  # Preprocess data for KMeans metric (used for Mahalanobis).
            D_centroids = self.kmeans.fit_predict(D_centroids)  # Perform KMeans clustering.

        D_centroids = self.apply_tukey(D_centroids)  # Apply Tukey transformation to centroids.
        if self.is_normalization:
            D_centroids = self.data_normalization(D_centroids)  # Normalize centroids if normalization is enabled.

        if self.is_normalization:
            self.D = self.data_normalization(self.D)  # Normalize the data if normalization is enabled.

        if self.is_first_fit:
            # On the first call: initialize parameters and store data
            self.is_first_fit = False
            self.D_centroids = D_centroids
            self.samples_per_class = D.size(1)
            self.n_features = D.size(2)
        else:
            # On subsequent calls: concatenate new centroids with existing data
            self.D_centroids = torch.concat((self.D_centroids, D_centroids))
        self.n_classes = self.D_centroids.size(0)

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
        if self.is_normalization:
            X = self.data_normalization(X)

        return self.metric.calculate_batch(self.model_predict, self.D_centroids, X, self.batch_size)

    @staticmethod
    def data_normalization(T, epsilon=1e-8):
        """ Normalizes the data tensor T """
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
    def accuracy_score(y_true, pred, verbose=False):
        """ Calculates the accuracy score. """

        if verbose:
            task_y_true = y_true // 10
            task_pred = pred // 10
            precision, recall, fscore, support = score(task_y_true.detach().cpu().numpy(),
                                                       task_pred.detach().cpu().numpy())

            tasks = sorted(set(task_y_true.detach().cpu().numpy()))
            # Create DataFrame for formatted output
            data = {
                "Task": tasks,
                "Precision": [f"{p * 100:.0f}%" for p in precision],
                "Recall": [f"{r * 100:.0f}%" for r in recall],
                "FScore": [f"{f:.2f}" for f in fscore],
                "% of all Answers": [f"{((task_pred == task).sum() + 1e-16) / len(task_pred) * 100:.2f}%"
                                     for task in tasks],
            }

            if verbose > 1:
                for i in range(10):
                    data.update({f"Class {i}": [f"{((pred == (10 * task + i)).sum() + 1e-16) / len(pred) * 100:.2f}%"
                                                for task in tasks]})

            df = pd.DataFrame(data)
            print(df.to_markdown(index=False))

        return torch.sum(torch.eq(y_true, pred)).item() / len(y_true) * 100
