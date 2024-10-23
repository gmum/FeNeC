import torch

import Metrics


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=0, metric=Metrics.EuclideanMetric()):
        """
        Initialize KMeans clustering algorithm.

        Parameters:
         n_clusters (int): The number of clusters to form.
         max_iter (int): Maximum number of iterations to run the algorithm.
         tol (float): Tolerance for stopping criteria (the algorithm stops if centroid movement is less than tol).
         metric (Metrics): The distance metric to use for calculating distances between points.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric

    def metric_preprocess(self, D):
        """ Preprocess metric: currently only used for Mahalanobis Metric """
        # We save the data for later (fit_predict method) and will preprocess the metric there
        self.D_preprocess = D

    def kmeans_plusplus(self, D):
        """
        Initialize centroids using the k-means++ algorithm.

        Parameters:
         D (torch.Tensor): Dataset tensor of shape [n_classes, samples_per_class, n_features].

        Returns:
         torch.Tensor: Initialized centroids of shape [n_classes, n_clusters, n_features].
        """
        n_classes, n_samples, n_features = D.shape
        centroids = [D[:, torch.randint(0, n_samples, (1,))]]  # Randomly select the first centroid

        for _ in range(self.n_clusters - 1):
            # Compute the minimum squared distances from the current centroids
            dists = torch.cat([torch.cdist(D, centroid) for centroid in centroids], dim=-1)
            dists = dists.min(dim=-1)[0] ** 2

            # Choose a new centroid based on distance-weighted sampling
            indices = torch.multinomial(dists, num_samples=1)
            centroids.append(D[torch.arange(D.size(0)), indices.flatten()].unsqueeze(1))

        return torch.stack(centroids, dim=1).squeeze(2)

    def fit_predict(self, D, init='k-means++', seed=42):
        """
        Initialize centroids either using k-means++ or randomly

        Parameters:
         D (torch.Tensor): Dataset tensor of shape [n_classes, samples_per_class, n_features].
         init (str): Initialization method ('k-means++' or 'random') for selecting initial centroids.
         seed (int): Random seed for initialization.

        Returns:
         torch.Tensor: Final centroids for each class of shape [n_classes, n_clusters, n_features].
        """
        # Set seed if applicable
        if seed != -1:
            torch.manual_seed(seed)

        if init == 'k-means++':
            centroids = self.kmeans_plusplus(D)
        elif init == 'random':
            random_indices = torch.randperm(D.size(1))[:self.n_clusters]
            centroids = D[:, random_indices]

        # Iterate over each class to calculate the centroids
        for d_class in range(D.size(0)):
            # In the case of Mahalanobis Distance for KMeans:
            # For each class, we reset the metric and calculate a new covariance matrix using only this class.
            # This covariance matrix will be used to calculate the distances
            # between the centroids and the data belonging to this class.
            if isinstance(self.metric, Metrics.MahalanobisMetric):
                self.metric.is_first_preprocess = True
                self.metric.preprocess(self.D_preprocess[d_class:d_class + 1])

            # Perform iterations up to max_iter (though it is usually less because of self.tol)
            for i in range(self.max_iter):
                # Calculate distances between points and centroids
                distances = self.metric.calculate(D[d_class].unsqueeze(0), centroids[d_class].unsqueeze(1))
                distances = distances.reshape(self.n_clusters, D.size(1))  # Adjust dimension if necessary

                # Assign points to the nearest centroid (cluster)
                cluster_labels = torch.argmin(distances, dim=0)
                # Calculate new centroids by averaging points in each cluster
                new_centroids = torch.stack([D[d_class, (cluster_labels == j)].mean(dim=0)
                                             for j in range(self.n_clusters)])

                # Handle NaN values in new centroids
                nans = torch.argwhere(torch.any(torch.isnan(new_centroids), axis=1)).flatten()
                for c in nans:
                    # Replace NaN centroid with previous one
                    new_centroids[c] = centroids[d_class, c]

                # Check if centroid updates are within tolerance to stop early
                if torch.max(torch.norm(new_centroids - centroids[d_class], dim=1)) <= self.tol:
                    break

                # Update centroids for the next iteration
                centroids[d_class] = new_centroids

        # Return the final centroids
        return centroids
