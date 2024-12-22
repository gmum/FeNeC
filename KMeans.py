import torch

import Metrics


class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=0, metric=Metrics.EuclideanMetric(), seed=42):
        """
        Initialize KMeans clustering algorithm.

        Parameters:
         - n_clusters (int): The number of clusters to form.
         - max_iter (int): Maximum number of iterations to run the algorithm.
         - tol (float): Tolerance for stopping criteria (the algorithm stops if centroid movement is less than tol).
         - metric (Metrics.Metric): The distance metric to use for calculating distances between points.
         - seed (int): Random seed for initializing centroids.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.metric = metric
        self.seed = seed

    def metric_preprocess(self, D):
        """ Preprocess metric: currently only used for Mahalanobis Metric """
        # We save the data for later (fit_predict method) and will preprocess the metric there
        self.D_preprocess = D.detach().clone()

    def kmeans_plusplus(self, D):
        """
        Initialize centroids using the k-means++ algorithm.

        Parameters:
         - D (torch.Tensor): Dataset tensor of shape [n_classes, samples_per_class, n_features].

        Returns:
         - torch.Tensor: Initialized centroids of shape [n_classes, n_clusters, n_features].
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

    def fit_predict(self, D, init='k-means++', seed=None):
        """
        Initialize centroids either using k-means++ or randomly

        Parameters:
         - D (torch.Tensor): Dataset tensor of shape [n_classes, samples_per_class, n_features].
         - init (str): Initialization method ('k-means++' or 'random') for selecting initial centroids.
         - seed (int): Random seed for initialization. If provided, this overrides the global seed set in __init__.

        Returns:
         - torch.Tensor: Final centroids for each class of shape [n_classes, n_clusters, n_features].
        """
        # Set random seed if specified
        seed = seed if seed is not None else self.seed
        if seed != -1:
            torch.manual_seed(seed)

        if init == 'k-means++':
            centroids = self.kmeans_plusplus(D)
        else:  # init == 'random'
            random_indices = torch.randperm(D.size(1))[:self.n_clusters]
            centroids = D[:, random_indices]

        # Iterate over each class to calculate the centroids
        for d_class in range(D.size(0)):
            # Preprocessing the metric on the current class (used in the case of Mahalanobis Distance for KMeans)
            if isinstance(self.metric, Metrics.MahalanobisMetric):
                self.metric.preprocess(self.D_preprocess[d_class:d_class + 1])

            # Perform iterations up to max_iter (though it is usually less because of self.tol)
            for i in range(self.max_iter):
                # Calculate distances between points and centroids and assign points to the nearest centroid (cluster)
                min_dist = lambda dists: (torch.argmin(dists.reshape(-1, self.n_clusters), dim=1))
                cluster_labels = self.metric.calculate_batch(min_dist, centroids[d_class].unsqueeze(0), D[d_class], -1)

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
