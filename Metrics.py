import abc

import torch


class Metric(abc.ABC):
    """ Abstract base class for a distance metric. """

    def preprocess(self, D):
        """ Optional method for preprocessing data (default is no-op). """
        pass

    @abc.abstractmethod
    def calculate(self, a, b):
        """ Abstract method to calculate the distance between two tensors 'a' and 'b'. """
        pass

    def reset(self):
        """ Method for resetting the state of the metric after preprocessing it. """
        pass

    def calculate_batch(self, fun, a, b, batch_size):
        """
        Calculates the batched distance between two tensors 'a' and 'b' and applies a function to the computed distances.

        Parameters:
         - fun (function): A function to apply to the computed distances.
         - a (torch.Tensor): The first tensor of shape [n_classes, samples_per_class, n_features].
         - b (torch.Tensor): The second tensor [n_samples_b, n_features].
         - batch_size (int): The batch size for tensor 'b'. If set to -1, it uses the full size of 'b'.

        Returns:
         - torch.Tensor: The results of the function 'fun' applied to the distances between 'a' and 'b'.
        """
        # Add a dimension so that it is of shape [1, n_classes, samples_per_class, n_features].
        if a.ndim == 3:
            a = a.unsqueeze(0)

        # Set batch size for 'b' to full size if specified as -1
        if batch_size == -1:
            batch_size = b.size(0)

        res = []

        for batch_b in b[:, None, None, :].split(batch_size, dim=0):
            # Compute distances between each batch of 'a' and 'batch_B', then reshape
            # Shape of distances: [batch_size, n_classes, samples_per_class]
            distances = self.calculate(a, batch_b).reshape(batch_b.size(0), a.size(1), a.size(2))

            # Apply the function to the calculated distances and append to results list
            res.append(fun(distances))

        # Concatenate all processed batches into a single tensor along the first dimension
        return torch.cat(res)


class EuclideanMetric(Metric):
    """ Computes the Euclidean distance between tensors. """

    def calculate(self, a, b):
        return torch.cdist(b.float(), a.float())


class CosineMetric(Metric):
    """ Computes the Cosine distance (1 - cosine similarity) between tensors. """

    def calculate(self, a, b):
        dot_product = torch.sum(a * b, dim=-1)
        norms_a = torch.norm(a.float(), p=2, dim=-1)
        norms_b = torch.norm(b.float(), p=2, dim=-1)
        res = 1 - dot_product / (norms_a * norms_b)
        return torch.clamp(res, 1e-30, 2)  # Ensure numerical stability


class MahalanobisMetric(Metric):
    """ Computes the Mahalanobis distance between tensors. """

    def __init__(self, shrinkage=0, gamma_1=1, gamma_2=1, normalization=True):
        """
        Initialize the Mahalanobis metric.

        Parameters:
         - shrinkage (int): Shrinkage type (0: none, 1: normal, 2: double).
         - gamma_1 (float): Diagonal shrinkage factor.
         - gamma_2 (float): Off-diagonal shrinkage factor.
         - normalization (bool): Whether to normalize the covariance matrix.
        """
        self.shrinkage = shrinkage
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        self.normalization = normalization
        self.is_first_preprocess = True

    def preprocess(self, D):
        """
        Preprocess the dataset to calculate and store the inverse covariance matrix.
        It can be called multiple times (is_first_preprocess=True only on the first call).
        On each call, the inverse covariance matrix is calculated for the new data
        and concatenated with the previous matrix.

        Parameters:
         - D (torch.Tensor): Dataset tensor of shape [n_classes, samples_per_class, n_features].
                             In subsequent calls, the samples_per_class and n_features dimensions
                             must match the initial call.
        """
        if self.is_first_preprocess:
            # First preprocess call: initialize parameters and calculate the inverse covariance matrix
            self.is_first_preprocess = False
            self.n_classes = D.size(0)
            self.samples_per_class = D.size(1)
            self.n_features = D.size(2)
            self.inv_cov_matrix = self.compute_inv_cov_matrix(D)
        else:
            # Subsequent calls: add new classes to the existing inverse covariance matrix
            self.n_classes += D.size(0)
            task_inv_cov_matrix = self.compute_inv_cov_matrix(D)
            # Concatenating new class covariance matrices along the first dimension (class axis)
            self.inv_cov_matrix = torch.concat((self.inv_cov_matrix, task_inv_cov_matrix), dim=0)

    def cov_matrix_shrinkage(self, D, cov_matrix):
        """ Apply shrinkage to the covariance matrix based on gamma_1 and gamma_2. """
        diag = cov_matrix.diagonal(dim1=1, dim2=2)

        # V1: Mean of the diagonal elements (variance) for each class
        V1 = diag.mean(1).reshape(-1, 1, 1).to(D.device)
        # V2: Mean of the off-diagonal elements (covariances) for each class
        V2 = ((cov_matrix.sum((1, 2)) - diag.sum(1)) / (self.n_features * (self.n_features - 1))).reshape(-1, 1, 1)
        # Id: Identity matrix repeated for each class
        Id = torch.eye(self.n_features).repeat(D.size(0), 1, 1).to(D.device)

        # Apply shrinkage to diagonal (using V1 and gamma_1) and off-diagonal (using V2 and gamma_2)
        return cov_matrix + self.gamma_1 * V1 * Id + self.gamma_2 * V2 * (1 - Id)

    def cov_matrix_normalization(self, cov_matrix):
        """ Normalize the covariance matrix based on standard deviations. """
        stds = torch.sqrt(self.cov_matrix.diagonal(dim1=1, dim2=2))
        return cov_matrix / torch.einsum('bi,bj->bij', stds, stds)

    def compute_inv_cov_matrix(self, D):
        """ Compute the covariance matrix and its inverse for each class. """
        cov_matrix = D - torch.mean(D, dim=1, keepdim=True)
        cov_matrix = torch.matmul(cov_matrix.transpose(1, 2), cov_matrix) / self.n_features

        if self.shrinkage:
            cov_matrix = self.cov_matrix_shrinkage(D, cov_matrix)
            if self.shrinkage == 2:
                cov_matrix = self.cov_matrix_shrinkage(D, cov_matrix)

        if self.normalization:
            self.cov_matrix = cov_matrix
            cov_matrix = self.cov_matrix_normalization(cov_matrix)

        # Return the inverse of the covariance matrix (of shape [n_classes, n_features, n_features])
        return torch.linalg.pinv(cov_matrix)

    def calculate(self, a, b):
        """
        Calculate the squared Mahalanobis distance between tensors a and b. The number of classes in tensor 'a' may
        differ from the number of classes in 'self.inv_cov_matrix'. In that case, the Mahalanobis distance shall be
        calculated using only the last 'a_n_classes'.

        Parameters:
         - a (torch.Tensor): First tensor of shape [1, a_n_classes, n_samples_a, n_features].
         - b (torch.Tensor): Second tensor of shape [n_samples_b, 1, 1, n_features].

        Returns:
         - torch.Tensor: Mahalanobis distance between a and b. Shape [n_samples_b, a_n_classes, n_samples_a].
        """

        # Compute the Mahalanobis distance
        diff = b - a  # [n_samples_b, a_n_classes, n_samples_a, n_features]
        return torch.einsum('abcd,bed,abce->abc', diff, self.inv_cov_matrix[-a.size(1):], diff)

    def reset(self):
        self.is_first_preprocess = True
