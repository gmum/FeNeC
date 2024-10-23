import torch


class KNNClassifier:
    def __init__(self, n_neighbors, metric, is_normalization=True, tukey_lambda=1., kmeans=None, device='cpu'):
        """
        Initializes the KNNClassifier.

        Parameters:
         n_neighbors (int): Number of nearest neighbors to consider.
         metric (Metric): A metric object to calculate the distance between points.
         is_normalization (bool): Whether to normalize the data.
         tukey_lambda (float): Lambda value for Tukey’s Ladder of Powers transformation.
         kmeans (KMeans): Optional k-means object for clustering.
         device (str): Device on which to run computations (e.g., 'cpu' or 'cuda').
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.is_normalization = is_normalization
        self.tukey_lambda = tukey_lambda
        self.kmeans = kmeans
        self.device = device
        self.D = None
        self.n_classes = None
        self.samples_per_class = None
        self.is_first_fit = True

    def apply_tukey(self, T):
        """ Applies Tukey’s Ladder of Powers transformation to the tensor T. """
        if self.tukey_lambda != 0:
            return torch.pow(T, self.tukey_lambda)
        else:
            return torch.log(T)

    def fit(self, D):
        """
        Fits the model to the training data.
        It can be called multiple times (is_first_fit=True only on the first call).
        On each call, the new data (new classes) is processed and concatenated with the old one.

        Parameters:
         D (torch.Tensor): Training data tensor of shape [n_classes, samples_per_class, n_features].
                           In subsequent calls, the samples_per_class and n_features dimensions
                           must match the initial call.
        """
        if D.ndim == 2:
            D = D.unsqueeze(0)  # Ensure data has the correct shape.
        D = D.type(torch.float32).to(self.device)

        # Processing the data:

        # Clone the dataset before applying Tukey for later usage in kmeans
        D_kmeans = D.clone()

        D = self.apply_tukey(D)  # Apply Tukey transformation.
        self.metric.preprocess(D)  # Preprocess the data for the distance metric.

        if self.kmeans is not None:
            self.kmeans.metric_preprocess(D)  # Preprocess data for k-means metric (used for Mahalanobis Metric).
            D = self.kmeans.fit_predict(D_kmeans)  # Perform k-means clustering.

        D = self.apply_tukey(D)  # Apply Tukey transformation after clustering.
        if self.is_normalization:
            D = self.data_normalization(D)  # Normalize the data

        if self.is_first_fit:
            # First preprocess call: initialize model parameters and store fitted data
            self.is_first_fit = False
            self.D = D
            self.n_classes = self.D.size(0)
            self.samples_per_class = self.D.size(1)
            self.n_features = self.D.size(2)
        else:
            # Subsequent calls: concatenate with existing data.
            self.D = torch.concat((self.D, D))
            self.n_classes = self.D.size(0)

        return self

    def predict(self, X, batch_size_X=1, batch_size_D=-1):
        """
        Predicts the class labels for the input data X.

        Parameters:
         X (torch.Tensor): Test data tensor of shape [n_samples, n_features].
         batch_size_X (int): Batch size for splitting test data (default: 1).
         batch_size_D (int): Batch size for splitting training data (default: -1, which means no splitting).

        Returns:
         torch.Tensor: Predicted class labels for the test data.
        """
        # Process the test data
        X = self.apply_tukey(X)
        if self.is_normalization:
            X = self.data_normalization(X)

        # Set default batch sizes if necessary
        if batch_size_X == -1:
            batch_size_X = X.size(0)
        if batch_size_D == -1:
            batch_size_D = self.D.size(1)

        pred = []  # List to store predictions.
        split_D = self.D[None, :, :, :].split(batch_size_D, dim=2)  # Split training data to speed up processing.

        # Iterate over batches of test samples.
        for batch_X in X[:, None, None, :].split(batch_size_X, dim=0):
            # Calculate the distances between the test sample and all training samples
            distances = torch.cat([self.metric.calculate(batch_D, batch_X) for batch_D in split_D],
                                  dim=-1).reshape(batch_X.size(0), -1)

            # Get the distances and indices of the k closest training samples
            values, indices = torch.topk(distances, self.n_neighbors, sorted=False, largest=False)

            # Calculate the class with the highest weighted vote
            classes = torch.zeros((batch_X.size(0), self.n_classes), dtype=torch.float32, device=self.device)
            classes.scatter_add_(1, indices // self.samples_per_class, 1. / values)
            pred.append(classes.argmax(1))  # Append predicted class with most votes.

        return torch.cat(pred)  # Return concatenated predictions.

    @staticmethod
    def data_normalization(T, epsilon=1e-8):
        """ Normalizes the data """
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
         X (torch.Tensor): Data tensor of shape [n_samples, n_features].
         y (torch.Tensor): Labels tensor of shape [n_samples].

        Returns:
         torch.Tensor: Transformed tensor D of shape [n_classes, samples_per_class, n_features].
        """
        n_classes = len(torch.unique(y))  # Determine number of unique classes.
        D = [[] for _ in range(n_classes)]  # Create list for each class.
        for _X, _y in zip(X, y):
            D[_y].append(_X)  # Group samples by their class.

        # Trim all classes to have the same number of samples.
        min_len = min([len(d) for d in D])
        D = [d[:min_len] for d in D]  # Ensure uniform number of samples per class.
        return torch.stack([torch.stack(D[i]) for i in range(n_classes)]).type(torch.float32).to(X.device)

    @staticmethod
    def accuracy_score(y_true, pred):
        """ Calculates the accuracy score. """
        return torch.sum(torch.eq(y_true, pred)).item() / len(y_true) * 100
