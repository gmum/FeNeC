import torch

import Metrics
from Classifier import Classifier
from KMeans import KMeans


class MLPClassifier(Classifier):
    def __init__(self, model, n_store=50, selection_method='random', metric_kmeans=Metrics.EuclideanMetric(), *args,
                 **kwargs):
        """
        Initializes the MLPClassifier.

        Parameters:
         - model (object): A classifier model that has the functions fit() and predict() (like LogisticRegression).
         - n_store (int): Number of samples from each class in the current task to retain for the next tasks.
         - selection_method (str): Method for selecting samples. Options:
            * 'all': Retains all samples.
            * 'random': Retains a random selection of samples.
            * 'kmeans': Retains centroid-based samples using KMeans.
         - metric_kmeans (Metric.Metric): Optional metric used for KMeans clustering when selection_method is 'kmeans'.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.n_store = n_store
        self.selection_method = selection_method
        self.metric_kmeans = metric_kmeans
        self.D_samples = None

    def model_predict(self, distances):
        """
        Method for predicting class labels based on distances to training samples.

        Parameters:
         - distances (Tensor): Distances of shape [batch_size, n_classes, n_centroids].

        Returns:
         - Tensor: Predicted class labels.
        """
        values = self.min_dist(distances)
        # Use the logistic regression model to predict the class based on these minimum distance values
        return torch.tensor(self.model.predict(values.cpu()), dtype=torch.float32, device=distances.device)

    def fit(self, D, train=True, **kwargs):
        """
        Trains the classifier on the current task data.

        Parameters:
         - D (torch.Tensor): Training data for the current task. Used solely for passing to the base classifier.
         - train (bool): If True, trains the classifier; if False, only prepares data without training.
        """
        super().fit(D)  # `self.D` (current task data) and `self.D_centroids` (class centroids) are now accessible.

        # Select up to n_store samples per class according to the selection method (unless it's 'all').
        if self.selection_method == 'random':
            # Randomly select n_store samples for each class
            indices = torch.stack([torch.randperm(self.n_store) for i in range(self.D.size(0))]).to(self.D.device)
            D_samples = self.D[torch.arange(self.D.size(0)).unsqueeze(1), indices]
        elif self.selection_method == 'kmeans':
            # Use KMeans clustering to select representative samples if 'kmeans' method is specified
            kmeans = KMeans(n_clusters=self.n_store, metric=self.metric_kmeans)
            kmeans.metric_preprocess(self.D)
            D_samples = kmeans.fit_predict(self.D)
        else:
            # Use all samples if 'all' or an invalid selection method is specified
            D_samples = self.D

        # Append the new samples to D_samples, which accumulates data samples across tasks
        self.D_samples = D_samples if self.D_samples is None else torch.cat((self.D_samples, D_samples), dim=0)

        if train:
            # Compute distances between all samples from previous tasks (self.D_samples)
            #  and class centroids (self.D_centroids)
            X = self.metric.calculate_batch(self.min_dist, self.D_centroids,
                                            self.D_samples.flatten(0, 1), self.batch_size)

            # Generate labels for the classifier based on the number of classes
            y = torch.tensor([[i] * D_samples.size(1) for i in range(self.n_classes)], dtype=torch.float32).flatten()

            # Train a classifier model on the distance-based features
            self.model.fit(X.cpu(), y.cpu())

    @staticmethod
    def min_dist(distances):
        """
        Computes the minimum distance to the closest centroid for each class.

        Used to prepare input data for LogisticRegression, where each sample is represented
        by its closest distance to a centroid for each class.

        Parameters:
         - distances (torch.Tensor): Tensor of shape [batch_size, n_classes, n_centroids].

        Returns:
         - torch.Tensor: Minimal distances of shape [batch_size, n_classes].
        """
        return distances.min(-1)[0]
