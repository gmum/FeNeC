import time
import unittest

import numpy as np
import scipy
import sklearn
import torch
import torchvision
from sklearn import datasets, neighbors, cluster

import Metrics
from KMeans import KMeans
from KNNClassifier import KNNClassifier


class TestMetrics(unittest.TestCase):
    def test_euclidean_calculate(self):
        """ Test the Euclidean distance calculation with 3 features. """
        a = torch.tensor([[[2, 3, 4], [3, 4, 0]],
                          [[0, -3, -6], [-2, 6, 4]]])
        b = torch.tensor([[2, 3, 4]], dtype=torch.float32)

        metric = Metrics.EuclideanMetric()

        dist_test = metric.calculate(a, b).squeeze()
        dist_correct = torch.tensor([[0, torch.sqrt(torch.tensor(18))], [torch.sqrt(torch.tensor(140)), 5]])

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))

    def test_cosine_calculate(self):
        """ Test the Cosine distance calculation using sklearn function. """
        a = torch.rand((100, 3000), dtype=torch.float32)
        b = torch.rand(3000, dtype=torch.float32)

        metric = Metrics.CosineMetric()

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor(sklearn.metrics.pairwise.cosine_distances(a, b.reshape(1, -1)),
                                    dtype=torch.float32).flatten()

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-5))

    def test_mahalanobis_distance(self):
        D = torch.tensor([[[1, 2], [2, 3], [3, 5]], [[5, 7], [4, -3], [10, 0]]], dtype=torch.float32)
        a = torch.mean(D, 1).unsqueeze(1)
        b = torch.tensor([[2, 4]])

        metric = Metrics.MahalanobisMetric(normalization=False)
        metric.preprocess(D)

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor([[[5.33336], [1.99815]]])

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))

    def test_mahalanobis_distance_2(self):
        D = torch.tensor([[[1, 2], [2, 3], [3, 5]], [[5, 7], [4, -3], [10, 0]]], dtype=torch.float32)
        a = torch.mean(D, 1).unsqueeze(1)
        b = torch.tensor([[2, 4], [-3, 0], [1, 2]])

        metric = Metrics.MahalanobisMetric(normalization=False)
        metric.preprocess(D)

        dist_test = metric.calculate(a, b)
        dist_correct = torch.tensor([[[5.33336], [1.99815]], [[233.33333], [8.64758]], [[1.33333], [2.75284]]])

        self.assertTrue(torch.allclose(dist_test, dist_correct, atol=1e-4))


class TestKNNClassifier(unittest.TestCase):
    def test_get_d(self):
        X = torch.tensor([[3, 2], [-5, 3], [2, 0], [4, 2], [-1, -1]])
        y = torch.tensor([0, 1, 1, 0, 0])
        D = torch.tensor([[[3, 2], [4, 2]], [[-5, 3], [2, 0]]])

        self.assertTrue(torch.equal(KNNClassifier.getD(X, y), D))

    def moons_sklearn_helper(self, metric, metric_sklearn):
        """ Helper function to test the KNN classifier with the moons dataset against sklearn's implementation. """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create a moons dataset
        train_samples, test_samples = 2000, 1000
        noise = 0.2
        n_neighbors = 1

        X_train, y_train = datasets.make_moons(n_samples=train_samples, noise=noise)
        X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train)
        X_test, y_test = datasets.make_moons(n_samples=test_samples, noise=noise)
        X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test)

        # Transform X_train and y_train into D for my KNN
        D = KNNClassifier.getD(X_train, y_train).to(device)

        # Transform D into X_train and y_train for sklearn KNN (so that it has the same samples as my KNN)
        X_train = D.reshape(-1, 2).cpu()
        y_train = torch.tensor([[i] * D.size(1) for i in range(2)]).flatten()

        # Initialize my KNN and predict classes of some test samples
        start = time.time()
        knn1 = KNNClassifier(n_neighbors=n_neighbors, metric=metric, device=device).fit(D)
        pred1 = knn1.predict(X_test.to(device))
        print('my knn: ', time.time() - start, KNNClassifier.accuracy_score(y_test.to(device), pred1))

        if metric_sklearn != 'mahalanobis':
            # Initialize sklearn KNN and predict classes of some test samples
            start = time.time()
            knn2 = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', algorithm='brute',
                                                  metric=metric_sklearn, n_jobs=-1).fit(X_train, y_train)
            pred2 = torch.tensor(knn2.predict(X_test), dtype=torch.int32)
            print('sklearn knn: ', time.time() - start, sklearn.metrics.accuracy_score(y_test, pred2) * 100)

            self.assertTrue(torch.equal(pred1.cpu(), pred2))

    def test_moons_sklearn_euclidean(self):
        self.moons_sklearn_helper(Metrics.EuclideanMetric(), 'euclidean')

    """
    Small differences in floating-point values can lead to an excessively large impact.
    def test_moons_sklearn_cosine(self):
        self.moons_sklearn_helper(Metrics.CosineMetric(), 'cosine')
    """

    def test_moons_sklearn_mahalanobis(self):
        self.moons_sklearn_helper(Metrics.MahalanobisMetric(True, True), 'mahalanobis')

    def mnist_sklearn_helper(self, metric, metric_sklearn):
        """ Helper function to test the KNN classifier with the MNIST dataset against sklearn's implementation. """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Due to how sklearn breaks ties, n_neighbors must be equal to 1 to get the same results.
        n_neighbors = 1

        # Download MNIST dataset
        train = torchvision.datasets.MNIST('/files/', train=True, download=True)
        X_train = train.data.reshape(-1, 28 * 28).type(torch.float32)
        y_train = train.targets

        test = torchvision.datasets.MNIST('/files/', train=False, download=True)
        X_test = test.data.reshape(-1, 28 * 28).type(torch.float32)[:1000]
        y_test = test.targets[:1000]

        # Transform X_train and y_train into D for my KNN (which requires the same amount of samples per each class)
        D = KNNClassifier.getD(X_train, y_train).to(device)

        # Transform D into X_train and y_train for sklearn KNN (so that it has the same samples as my KNN)
        X_train = D.reshape(-1, 28 * 28).cpu()
        y_train = torch.tensor([[i] * D.size(1) for i in range(10)]).flatten()

        # Initialize my KNN and predict classes of some test samples
        start = time.time()
        knn1 = KNNClassifier(n_neighbors=n_neighbors, metric=metric, device=device).fit(D)
        pred1 = knn1.predict(X_test.to(device))
        print('my knn: ', time.time() - start, KNNClassifier.accuracy_score(y_test.to(device), pred1))

        if metric_sklearn != 'mahalanobis':
            # Initialize sklearn KNN and predict classes of some test samples
            start = time.time()
            knn2 = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric=metric_sklearn,
                                                  algorithm='brute').fit(X_train, y_train)
            pred2 = torch.tensor(knn2.predict(X_test), dtype=torch.int32)
            print('sklearn knn: ', time.time() - start, sklearn.metrics.accuracy_score(y_test, pred2) * 100)

            self.assertTrue(torch.equal(pred1.cpu(), pred2))

    def test_mnist_sklearn_euclidean(self):
        self.mnist_sklearn_helper(Metrics.EuclideanMetric(), 'euclidean')

    def test_mnist_sklearn_cosine(self):
        self.mnist_sklearn_helper(Metrics.CosineMetric(), 'cosine')

    def test_mnist_sklearn_mahalanobis(self):
        self.mnist_sklearn_helper(Metrics.MahalanobisMetric(True, True), 'mahalanobis')

    def test_kmeans(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        X = torch.tensor(sklearn.datasets.make_blobs(n_samples=10000, n_features=2, centers=15, cluster_std=0.5)[0])

        # Alternatively:
        # test = torchvision.datasets.MNIST('/files/', train=False, download=True)
        # X = test.data.reshape(-1, 28 * 28).to(torch.float32)

        similarities = []
        for _ in range(50):
            centroids1 = KMeans(15).fit_predict(X.to(device).unsqueeze(0)).cpu().squeeze(0)
            centroids2 = torch.tensor(sklearn.cluster.KMeans(15).fit(X).cluster_centers_)

            distances = Metrics.EuclideanMetric().calculate(centroids1, centroids2)
            similarity_matrix = 1 - (distances / distances.max())

            row_indices, col_indices = scipy.optimize.linear_sum_assignment(similarity_matrix, maximize=True)
            similarity = similarity_matrix[row_indices, col_indices].mean()

            similarities.append(similarity.item())

        mean_similarity = np.array(similarities).mean()
        print("My KMeans similarity to sklearn's one:", mean_similarity)
        self.assertTrue(mean_similarity > 0.9)


if __name__ == '__main__':
    unittest.main()
