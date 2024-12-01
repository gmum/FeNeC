import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from Classifier import Classifier


def save_task_data(task, epoch, loss, alpha, a, b, r, filename="data.csv"):
    """
    Save task data to a CSV file. Creates the file if it doesn't exist.

    Parameters:
        task (str): The name of the task.
        epoch (int): The current epoch.
        alpha, a, b, r (torch.nn.Parameter): Torch parameters, their size can vary between tasks.
        filename (str): The name of the CSV file to save data to (default is "data.csv").
    """

    filename = str(task) + "_" + filename
    # Ensure inputs are torch.nn.Parameter and convert them to flat lists
    for param_name, param in zip(["alpha", "a", "b", "r"], [alpha, a, b, r]):
        if not isinstance(param, torch.nn.Parameter):
            raise TypeError(f"{param_name} must be of type torch.nn.Parameter, but got {type(param)}")

    # Convert tensor data to lists
    alpha_list = alpha.detach().numpy().tolist()
    a_list = a.detach().numpy().tolist()
    b_list = b.detach().numpy().tolist()
    r_list = r.detach().numpy().tolist()

    # Prepare data row
    row = [task, epoch, loss] + alpha_list + a_list + b_list + r_list

    # Check if the file exists and determine if a header is needed
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if file is being created
        if not file_exists:
            header = (
                    ["task", "epoch", "loss"] +
                    [f"alpha_{i}" for i in range(len(alpha_list))] +
                    [f"a_{i}" for i in range(len(a_list))] +
                    [f"b_{i}" for i in range(len(b_list))] +
                    [f"r_{i}" for i in range(len(r_list))]
            )
            writer.writerow(header)

        # Write data row
        writer.writerow(row)


class GradKNNClassifier(Classifier):
    def __init__(self, n_points=10, mode=0, num_epochs=100, kmeans=None, lr=1e-3, early_stop_patience=10,
                 reg_type=1, reg_lambda=0.1, verbose=True, *args, **kwargs):
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
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.verbose = verbose
        self.parameters = torch.nn.ParameterDict()
        if self.mode == 0:
            for parameter_name in ['alpha', 'a', 'b']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.randn(1, device=self.device))})
        else:
            for parameter_name in ['alpha', 'a', 'b', 'r']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.empty(0, device=self.device))})
            self.original_parameters = self.parameters.copy()

    def net_predict(self, data):
        if self.mode == 0:
            data_transformed = self.parameters['a'] + self.parameters['b'] * torch.log(data + 1e-16)
            data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
            data_sum = data_activated.sum(dim=-1)
            logits = self.parameters['alpha'] * data_sum
            return logits
        else:
            data_transformed = (self.parameters['a'][None, :, None] + self.parameters['b'][None, :, None]
                                * torch.log(data + 1e-16))
            data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
            data_sum = data_activated.sum(dim=-1)
            logits = self.parameters['alpha'][None, :] * data_sum + self.parameters['r'][None, :]
            return logits

    def n_nearest_points(self, distances):
        # TODO: zmienić opis
        # Funkcja do użycia w tworzeniu datasetu (i chyba przy predict)
        #  - znajduje self.n_points najbliższych punktów z każdej klasy
        # Wymiar distances: [batch_size, n_classes, samples_per_class (czyli przy użyciu kmeansów: n_centroids)]
        # Czyli chcemy znaleźć dla każdego z 'batch_size' punktów dla każdej klasy: n_points najbliższych sąsiadów
        # Czy powinny być posortowane te punkty? ma to jakieś znacznie? Podobnie w model_predict
        return torch.topk(distances, self.n_points, sorted=True, largest=False)[0]
        # Return shape: [batch_size, self.n_classes, self.n_points]

    def save_original_parameters(self):
        # Save original parameters for later use in calculating the regularization
        if self.mode == 1:
            for parameter in ['alpha', 'a', 'b', 'r']:
                self.original_parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.original_parameters[parameter], self.parameters[parameter][-self.D.size(0):]], dim=0))

    def regularization(self):
        if self.mode == 1:
            prev_num_classes = len(self.original_parameters['a'])
            for parameter in ['alpha', 'a', 'b', 'r']:
                diff = (self.parameters[parameter][:prev_num_classes]
                        - self.original_parameters[parameter])
                if self.reg_type == 1:
                    return self.reg_lambda * torch.sum(torch.abs(diff))
                else:
                    return self.reg_lambda * torch.sum(diff ** 2)
        return 0

    def fit(self, D):
        """
        Trains the classifier on the current task data.

        Parameters:
         - D (torch.Tensor): Training data for the current task. Used solely for passing to the base classifier.
        """
        super().fit(D)
        # Access to self.D (data from the current task) and
        #  self.D_centroids (class centroids from all tasks) is now available

        # Create the new parameters to train (or in the case of mode 0 and consequential task, use previous ones)
        if self.mode == 1:
            for parameter in ['alpha', 'a', 'b', 'r']:
                self.parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.parameters[parameter], torch.nn.Parameter(torch.randn(self.D.size(0), device=self.device))],
                    dim=0))

        # Prepare the DataLoader
        D_clip = -self.D.size(0) if self.mode == 0 else 0
        X = torch.cat([self.metric.calculate_batch(
            self.n_nearest_points,
            self.D_centroids[D_clip:],
            d_class,
            self.batch_size)
            for d_class in self.D])  # Shape: [points in the current task, classes in the current task, self.n_points]
        y = torch.arange(self.D.size(0)) if self.mode == 0 \
            else torch.arange(self.D_centroids.size(0) - self.D.size(0), self.D_centroids.size(0))
        y = y.repeat_interleave(self.D.size(1)).to(self.device)
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
        if self.verbose:
            print("Dataloader created")

        # Train the model
        best_validation_loss = float('inf')
        epochs_no_improve = 0
        optimizer = torch.optim.Adam(self.parameters.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # Track total loss for the epoch
            correct = 0  # Track correct predictions
            total = 0  # Track total samples

            for data, target in train_dataloader:
                # Forward pass
                predictions = self.net_predict(data)
                loss = F.cross_entropy(predictions, target)
                loss += self.regularization()

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
                    predictions = self.net_predict(data)
                    predicted_classes = torch.argmax(predictions, dim=1)
                    valid_correct += (predicted_classes == target).sum().item()
                    valid_total += target.size(0)
                    valid_loss += F.cross_entropy(predictions, target).item()
            avg_valid_loss = valid_loss / len(valid_dataloader)
            valid_accuracy = 100.0 * valid_correct / valid_total

            if self.verbose:
                # Save task data (TODO)
                # save_task_data(self.D_centroids.size(0), epoch, avg_valid_loss, parameters)
                if epoch % 20 == 0:
                    print(f"Validation Accuracy after Epoch [{epoch + 1}/{self.num_epochs}]: {valid_accuracy:.2f}%, "
                          f"Loss = {valid_loss / len(valid_dataloader):.4f},")

            if avg_valid_loss < best_validation_loss:
                # torch.save(parameters.state_dict(), "parameters.pth") # TODO
                # save_task_data(self.D_centroids.size(0), epoch, avg_valid_loss, parameters)
                best_validation_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                # parameters.load_state_dict(torch.load("parameters.pth"))
                self.save_original_parameters()
                return
        self.save_original_parameters()

    def model_predict(self, distances):
        with torch.no_grad():
            return torch.argmax(self.net_predict(self.n_nearest_points(distances)), -1)
