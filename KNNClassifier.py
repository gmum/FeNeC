import torch

from Classifier import Classifier


class KNNClassifier(Classifier):
    def __init__(self, n_neighbors, *args, **kwargs):
        """
        Initializes the K-Nearest Neighbors (KNN) Classifier.

        Parameters:
         - n_neighbors (int): Number of nearest neighbors to consider for classification.
        """
        super().__init__(*args, **kwargs)
        self.n_neighbors = n_neighbors

    def model_predict(self, distances):
        # Get the distances and indices of the k closest training samples
        values, indices = torch.topk(distances.flatten(1, 2), self.n_neighbors, sorted=False, largest=False)

        # Calculate the class with the highest weighted vote:
        # Initialize a tensor to store class scores for each test sample
        classes = torch.zeros((distances.size(0), distances.size(1)), dtype=torch.float32, device=distances.device)
        # Aggregate votes for each class by adding the inverse of distances to corresponding class scores
        classes.scatter_add_(1, indices // distances.size(2), 1. / values)

        # Return the class with the highest aggregated score for each test sample
        return classes.argmax(1)
