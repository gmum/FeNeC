import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from Classifier import Classifier


class GradKNNClassifier(Classifier):
    def __init__(self, n_points=10, mode=0, num_epochs=100, kmeans=None, *args, **kwargs):
        """
        Initializes the GradKNNClassifier.

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
        self.kmeans = kmeans
        # TODO: jakaś przykładowa nazwa, nie wiem jak najlepiej trzymać te parametry modelu
        self.parameters = None

    def net_predict(self, data, parameters=None):
        if parameters is None:
            parameters = self.parameters

        if self.mode == 0:
            alpha, a, b = parameters
            data_log = torch.log(data + 1e-16)

            data_transformed = a + b * data_log
            data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
            data_sum = data_activated.sum(dim=-1)
            logits = alpha * data_sum

            return logits
        else:  # self.mode == 1
            alpha, a, b, r = parameters
            data_log = torch.log(data + 1e-8)

            #print(a.shape, b.shape, data_log.shape, " shapes")
            data_transformed = a[None, :, None] + b[None, :, None] * data_log
            data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
            data_sum = data_activated.sum(dim=-1)
            logits = alpha[None, :] * data_sum + r[None, :]

            return logits

    def n_nearest_points(self, distances):
        # Funkcja do użycia w tworzeniu datasetu (i chyba przy predict)
        #  - znajduje self.n_points najbliższych punktów z każdej klasy
        # Wymiar distances: [batch_size, n_classes, samples_per_class (czyli przy użyciu kmeansów: n_centroids)]
        # Czyli chcemy znaleźć dla każdego z 'batch_size' punktów dla każdej klasy: n_points najbliższych sąsiadów
        # Czy powinny być posortowane te punkty? ma to jakieś znacznie? Podobnie w model_predict
        return torch.topk(distances, self.n_points, sorted=True, largest=False)[0]
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
        parameters = []  # Nie wiem czy to jest optymalne, żeby to była zwykła tablica? Może torch.empty(0)

        if self.mode == 0:
            # Pierwszy raz wywołujemy fit (pierwszy task)
            # TODO: (tylko przykład) zmienić to i w jakiś mądry sposób pewnie wybrać te początkowe
            #  (chyba są funkcje w pytorch do tego) !!!!!! teraz zawsze jest -1, 0 i 1, chyba tak nie chcemy
            if self.parameters is None:        
                alpha_param = torch.nn.Parameter(torch.tensor(-1., device=self.device))
                a_param = torch.nn.Parameter(torch.tensor(0., device=self.device))
                b_param = torch.nn.Parameter(torch.tensor(1., device=self.device))
                parameters = [alpha_param, a_param, b_param]
            else:
                parameters = self.parameters
        if self.mode == 1:
            # TODO: zmienić to na torch.Parameter i jakoś lepiej, masz dokładnie to samo w if i else
            a_param = torch.nn.Parameter(torch.zeros(self.D.size(0), device=self.device))
            b_param = torch.nn.Parameter(torch.ones(self.D.size(0), device=self.device))
            alpha_param = torch.nn.Parameter(torch.full((self.D.size(0),), -1.0, device=self.device))
            r_param = torch.nn.Parameter(torch.zeros(self.D.size(0), device=self.device))
            parameters = [alpha_param, a_param, b_param, r_param]


        # Prepare the DataLoader
        X = []
        for d_class in range(D.size(0)):
            X.append(self.metric.calculate_batch(self.n_nearest_points, self.D_centroids[-self.D.size(0):],
                                                 self.D[d_class], self.batch_size))
        X = torch.cat(X)  # Shape: [points in the current task, classes in the current task, self.n_points]
        y = torch.arange(self.D.size(0)).repeat_interleave(self.D.size(1)).to(self.device)
        dataset = TensorDataset(X, y)

        # Define the split ratio
        train_ratio = 0.9
        train_size = int(train_ratio * len(dataset))
        valid_size = len(dataset) - train_size

        # Split the dataset
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Create DataLoaders for training and validation
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        print("Dataloader created")

        # Train the model
        # TODO: Jako optimizer Adam, pewnie lepiej zrobić z tego hiperparametr


        best_validation_loss = float('inf') 
        epochs_no_improve = 0
        early_stop_patience = 10  # Number of epoch

        optimizer = torch.optim.Adam(parameters, lr=1e-3)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # Track total loss for the epoch
            correct = 0  # Track correct predictions
            total = 0  # Track total samples

            for data, target in train_dataloader:
                # Forward pass
                predictions = self.net_predict(data, parameters)
                loss = F.cross_entropy(predictions, target)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track loss
                epoch_loss += loss.item()

                # Calculate accuracy for the batch
                predicted_classes = torch.argmax(predictions, dim=1)
                correct += (predicted_classes == target).sum().item()
                total += target.size(0)

            # Calculate and display epoch metrics
            epoch_accuracy = 100.0 * correct / total
            valid_correct = 0
            valid_total = 0
            valid_loss = 0.0
            with torch.no_grad():
                for data, target in valid_dataloader:
                    predictions = self.net_predict(data, parameters)
                    predicted_classes = torch.argmax(predictions, dim=1)
                    valid_correct += (predicted_classes == target).sum().item()
                    valid_total += target.size(0)
                    
                    loss = F.cross_entropy(predictions, target)
                    valid_loss += loss.item()
            avg_valid_loss = valid_loss / len(valid_dataloader)
            valid_accuracy = 100.0 * valid_correct / valid_total
            
            
            if epoch % 20 == 0:
                print(f"Validation Accuracy after Epoch [{epoch + 1}/{self.num_epochs}]: {valid_accuracy:.2f}%, Loss = {valid_loss / len(valid_dataloader):.4f},")

            if avg_valid_loss < best_validation_loss:
                best_validation_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                # Save the model's parameters
                if self.mode == 0:
                    self.parameters = parameters
                elif self.mode == 1:
                    if self.parameters is None:
                        self.parameters = parameters
                    else:
                        self.parameters[0] = torch.cat((self.parameters[0], parameters[0]), dim=0)
                        self.parameters[1] = torch.cat((self.parameters[1], parameters[1]), dim=0)
                        self.parameters[2] = torch.cat((self.parameters[2], parameters[2]), dim=0)
                        self.parameters[3] = torch.cat((self.parameters[3], parameters[3]), dim=0)
                
                return
        if self.mode == 0:
            self.parameters = parameters
        elif self.mode == 1:
            if self.parameters is None:
                self.parameters = parameters
            else:
                self.parameters[0] = torch.cat((self.parameters[0], parameters[0]), dim=0)
                self.parameters[1] = torch.cat((self.parameters[1], parameters[1]), dim=0)
                self.parameters[2] = torch.cat((self.parameters[2], parameters[2]), dim=0)
                self.parameters[3] = torch.cat((self.parameters[3], parameters[3]), dim=0)

    def model_predict(self, distances):
        return torch.argmax(self.net_predict(self.n_nearest_points(distances)), -1)
