import torch

from Classifier import Classifier


class GradKNNClassifier(Classifier):
    def __init__(self, n_points=10, mode=0, num_epochs=100, *args, **kwargs):
        """
        Initializes the GradKNNClassifier.
        TODO: usunąć polskie komentarze i dodać ładne angielskie do tego co się dzieje w kodzie + dokumentacja funkcji
            + w pliku dataset_1.ipynb jest przykładowe użycie tej klasy (a przynajmniej na ten moment)

        Parameters:
         - n_points (int): Number of samples from each class in the current task to retain for the next tasks.
         - mode (int): The mode of the classifier:
            * 0: Train the same parameters for all classes.
            * 1: Train separate parameters for each class.
         - num_epochs (int): Number of epochs to train the classifier.
        """
        super().__init__(*args, **kwargs)
        self.n_points = n_points
        self.mode = mode
        self.num_epochs = num_epochs
        # TODO: jakaś przykładowa nazwa, nie wiem jak najlepiej trzymać te parametry modelu
        self.parameters = None

    def criterion(self, predictions, target):
        # TODO: zaimplementować (+ pewnie to będzie mogła być funkcja static)
        return None

    def predict(self, data, parameters=None):
        # TODO: zaimplementować (nie wiem jak z parameters, czy trzeba podawać? ciężko mi powiedzieć na tym etapie)
        # Pewnie najcięższa rzecz do zaimplementowania, trzeba napisać ten wzór z LeakyRelu
        # 'data' pewnie wymiaru [batch_size, self.n_classes, self.n_points]
        if parameters is None:
            parameters = self.parameters
        return None

    def n_nearest_points(self, distances):
        # Funkcja do użycia w tworzeniu datasetu (i chyba przy predict)
        #  - znajduje self.n_points najbliższych punktów z każdej klasy
        # Wymiar distances: [batch_size, n_classes, samples_per_class (czyli przy użyciu kmeansów: n_centroids)]
        # Czyli chcemy znaleźć dla każdego z 'batch_size' punktów dla każdej klasy: n_points najbliższych sąsiadów
        # Czy powinny być posortowane te punkty? ma to jakieś znacznie? Podobnie w model_predict
        return torch.stack([torch.topk(distances[:, d_class, :], self.n_points, sorted=True, largest=False)[0]
                            for d_class in range(self.n_classes)], dim=1)
        # Return shape: [batch_size, self.n_classes, self.n_points]

    def fit(self, D):
        """
        Trains the classifier on the current task data.

        Parameters:
         - D (torch.Tensor): Training data for the current task. Used solely for passing to the base classifier.
        """
        super().fit(D)
        # Access to self.D (data from the current task) and
        #  self.D_centroids (class centroids from all tasks) is now available

        # TODO: tylko taki generalny zarys

        parameters = []  # Nie wiem czy to jest optymalne, żeby to była zwykła tablica? Może torch.empty(0)

        if self.parameters is None:
            # Pierwszy raz wywołujemy fit (pierwszy task)
            # TODO: (tylko przykład) zmienić to i w jakiś mądry sposób pewnie wybrać te początkowe
            # (chyba są funkcje w pytorch do tego)
            alpha_param = torch.tensor(.3, requires_grad=True, device=self.device)
            a_param = torch.tensor(.6, requires_grad=True, device=self.device)
            b_param = torch.tensor(.9, requires_grad=True, device=self.device)
            parameters = [alpha_param, a_param, b_param]
        if self.mode == 1:
            # Pewnie coś podobnego: dla mode == 1 trzeba za każdym razem nowe parametry dla każdej klasy
            pass

        # Prepare the DataLoader
        # TODO: czyli po prostu wziąć self.n_points najbliżsych punktów każdej klasy dla każdego punktu z obecnego
        #  taska i powiedzieć jaka to klasa
        X, y = [], []
        for d_class in range(D.size(0)):
            X.append(self.metric.calculate_batch(self.n_nearest_points, self.D_centroids,
                                                 self.D[d_class], self.batch_size))
            y.append(torch.tensor([d_class] * self.D.size(1)))  # pewnie można zastąpić używając torch.repeat
        X, y = torch.cat(X), torch.cat(y)
        # Wymiar X: [liczba wszystkich punktów w tym tasku, self.n_classes (łączna liczba klas), self.n_points]
        # Wymiar y: [liczba wszystkich punktów w tym tasku]
        dataloader = None

        # Train the model
        # Jako optimizer Adam, pewnie lepiej zrobić z tego hiperparametr
        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        for epoch in range(self.num_epochs):
            for data, target in dataloader:
                # Forward pass
                predictions = self.predict(data, parameters)
                loss = self.criterion(predictions, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Zapis parametrów modelu
        if self.mode == 0:
            self.parameters = parameters
        elif self.mode == 1:
            if self.parameters is None:
                self.parameters = parameters
            else:
                self.parameters = torch.cat(self.parameters, parameters, dim=0)  # coś w tym stylu pewnie

    def model_predict(self, distances):
        # TODO: Użyć przetrenowanego modelu i zwrócić tensor zawierający [batch_size == distances.size(0)] klas
        return self.predict(self.n_nearest_points(distances))  # coś w tym stylu?
