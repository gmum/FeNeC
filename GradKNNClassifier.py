import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from Classifier import Classifier


def save_task_data(task, epoch, loss, parameters, filename="data.csv"):
    """
    Save task data to a CSV file. Appends to an existing file or creates a new one.

    Parameters:
        task (str): The name of the task.
        epoch (int): The current epoch.
        loss (float): Loss value for the epoch.
        alpha, a, b, r (torch.nn.Parameter): Torch parameters, their size can vary between tasks.
        filename (str): The name of the CSV file to save data to (default is "data.csv").
    """
    all_classes = 100  # Fixed size for parameter lists
    alpha = parameters["alpha"]
    a = parameters["a"]
    b = parameters["b"]
    r = parameters["r"]

    # Validate parameter types
    # for param_name, param in zip(["alpha", "a", "b", "r"], [alpha, a, b, r]):
    #    if not isinstance(param, torch.nn.Parameter):
    #        raise TypeError(f"{param_name} must be of type torch.nn.Parameter, but got {type(param)}")

    # Convert parameters to lists and pad to all_classes with zeros
    def pad_to_all_classes(param):
        param_list = param.detach().cpu().numpy().tolist()
        return param_list + [0] * (all_classes - len(param_list))

    alpha_list = pad_to_all_classes(alpha)
    a_list = pad_to_all_classes(a)
    b_list = pad_to_all_classes(b)
    r_list = pad_to_all_classes(r)

    # Prepare data row
    row = [task, epoch, loss] + alpha_list + a_list + b_list + r_list

    # Check if the file exists and determine if a header is needed
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header if the file does not exist
        if not file_exists:
            header = (
                    ["task", "epoch", "loss"] +
                    [f"alpha_{i}" for i in range(all_classes)] +
                    [f"a_{i}" for i in range(all_classes)] +
                    [f"b_{i}" for i in range(all_classes)] +
                    [f"r_{i}" for i in range(all_classes)]
            )
            writer.writerow(header)

        # Write the data row
        writer.writerow(row)


class GradKNNClassifier(Classifier):
    def __init__(self, n_points=10, mode=0, num_epochs=100, kmeans=None, lr=1e-3, early_stop_patience=10,
                 train_previous=True, reg_type=1, reg_lambda=0.1, use_sigmoid=False, sigmoidx=2, verbose=True,
                 when_norm=0, norm_type=0, add_prev_centroids=True, only_prev_centroids=False, repeat_prev_centroid=1,
                 optimizer_type=0, dataloader_batch_size=512, *args, **kwargs):
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
        self.train_previous = train_previous
        self.use_sigmoid = use_sigmoid
        self.sigmoidx = sigmoidx
        self.verbose = verbose
        self.parameters = torch.nn.ParameterDict()
        if self.mode == 0:
            for parameter_name in ['alpha', 'a', 'b']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.randn(1, device=self.device))})
        else:
            for parameter_name in ['alpha', 'a', 'b', 'r']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.empty(0, device=self.device))})
            self.original_parameters = self.parameters.copy()
        self.task_boundaries = torch.tensor([0])

        self.when_norm = when_norm
        self.norm_type = norm_type
        self.add_prev_centroids = add_prev_centroids
        self.repeat_prev_centroid = repeat_prev_centroid
        self.only_prev_centroids = only_prev_centroids
        self.optimizer_type = optimizer_type
        self.dataloader_batch_size = dataloader_batch_size

    def net_predict(self, data, normalize=True):
        parameters = self.parameters

        if self.mode == 0:
            data_transformed = parameters['a'] + parameters['b'] * torch.log(data + 1e-16)
            data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
            data_sum = data_activated.sum(dim=-1)
            logits = F.softplus(parameters['alpha']) * data_sum
            return logits
        else:
            if self.use_sigmoid:
                data_transformed = (torch.tanh(parameters['a'])[None, :, None] * self.sigmoidx +
                                    torch.tanh(parameters['b'])[None, :, None]  * self.sigmoidx * torch.log(data + 1e-16))
                data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
                data_sum = data_activated.sum(dim=-1)
                logits = F.softplus(torch.tanh(parameters['alpha'][None, :]) * self.sigmoidx) * data_sum + torch.tanh(parameters['r'][None, :]) * self.sigmoidx
            # Zobaczyc czy jak wstawimy na same logity zamiast wszystkich
            else:
                data_transformed = (parameters['a'][None, :, None] +
                                    parameters['b'][None, :, None] * torch.log(data + 1e-16))
                data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
                data_sum = data_activated.sum(dim=-1)
                logits = F.softplus(parameters['alpha'][None, :]) * data_sum + parameters['r'][None, :]

            if normalize or self.when_norm:
                if self.norm_type == 0:
                    normalized_logits = torch.cat(
                        [logits[:, start:end] / (1 + logits[:, start:end].std()) for start, end in
                         zip(self.task_boundaries[:-1], self.task_boundaries[1:])], dim=1)
                else:
                    normalized_logits = torch.cat(
                        [(logits[:, start:end] - logits[:, start:end].mean()) / logits[:, start:end].std() for
                         start, end in
                         zip(self.task_boundaries[:-1], self.task_boundaries[1:])], dim=1)
                return normalized_logits
            return logits

    def n_nearest_points(self, distances):
        # TODO: zmienić opis tej i pozostałych funkcji
        # Czy powinny być posortowane te punkty? ma to jakieś znacznie? Podobnie w model_predict
        # distances: [batch_size, n_classes, samples_per_class
        return torch.topk(distances, self.n_points, sorted=True, largest=False)[0]
        # Return shape: [batch_size, self.n_classes, self.n_points]

    def n_nearest_points_centroids(self, distances, class_num=-1):
        nearest_points = torch.topk(distances, self.n_points + 1, sorted=True, largest=False)[
            0]  # obliczam + 1 najbliższych centroidów
        nearest_points = torch.cat([nearest_points[:, :class_num, :-1],  # wywalam najdalszy z pozostałych klas
                                    nearest_points[:, class_num:(class_num + 1), 1:],  # wywalam najbliższy z tej klasy
                                    nearest_points[:, (class_num + 1):, :-1]], dim=1)
        return nearest_points

    def save_original_parameters(self):
        # Save original parameters for later use in calculating the regularization
        if self.mode == 1:
            for parameter in self.parameters:
                self.original_parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.original_parameters[parameter], self.parameters[parameter][-self.D.size(0):]], dim=0))

    def regularization(self):  # zapisac osobno loss cross entropy i regularization podczas kolejnych epok
        reg_loss = 0
        if self.mode == 1:
            prev_num_classes = len(self.original_parameters['a'])
            for parameter in self.parameters:
                diff = (self.parameters[parameter][:prev_num_classes]
                        - self.original_parameters[parameter])
                if self.reg_type == 1:
                    # L1 regularization
                    reg_loss += self.reg_lambda * torch.sum(torch.abs(diff))
                else:
                    # L2 regularization
                    reg_loss += self.reg_lambda * torch.sum(diff ** 2)
        return reg_loss

    losses_1, losses_reg = [], []

    def fit(self, D, task_num=-1, **kwargs):
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
            for parameter in self.parameters:
                new_param = torch.rand(self.D.size(0), device=self.device) - 0.5
                if parameter == 'alpha':
                    new_param = torch.abs(new_param)
                self.parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.parameters[parameter], torch.nn.Parameter(new_param)], dim=0))

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

        if self.mode == 1 and self.add_prev_centroids:  # z mode==0 pewnie nic nie zmieni, ale można sprawdzić
            D_range = self.D_centroids.size(0) if self.only_prev_centroids is False \
                else self.D_centroids.size(0) - self.D.size(0)  # wszystkie centroidy albo tylko z poprzednich klas
            if D_range is not 0:
                X_centroids = torch.cat([self.metric.calculate_batch(
                    lambda distances: self.n_nearest_points_centroids(distances, class_num),
                    # funkcja licząca najbliższe punkty do aktualnej klasy class_num
                    self.D_centroids,
                    self.D_centroids[class_num],
                    self.batch_size)
                    for class_num in range(D_range)])
                X = torch.cat([X, X_centroids.repeat(self.repeat_prev_centroid, 1, 1)])  # powtórzenie kilka razy

                y_centroids = (torch.arange(D_range).repeat_interleave(self.D_centroids.size(1)).to(self.device))
                y = torch.cat([y, y_centroids.repeat(self.repeat_prev_centroid)])

        dataset = TensorDataset(X, y)

        self.task_boundaries = torch.cat([self.task_boundaries, torch.tensor(X.size(1)).unsqueeze(0)])

        # Define the split ratio
        train_ratio = 0.9
        train_size = int(train_ratio * len(dataset))
        valid_size = len(dataset) - train_size

        # Split the dataset
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

        # Create DataLoaders for training and validation
        train_dataloader = DataLoader(train_dataset, batch_size=self.dataloader_batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.dataloader_batch_size, shuffle=False)
        if self.verbose:
            print("Dataloader created")
        self.losses_1.append([])
        self.losses_reg.append([])

        x = self.D_centroids.size(0) - self.D.size(0)  # Number of elements to freeze in each parameter

        # Register hooks to freeze the first `x` elements
        if not self.train_previous:
            for name, param in self.parameters.items():
                def hook_fn(grad, num_freeze=x):
                    # Zero out the gradient for the first `x` elements
                    grad[:num_freeze] = 0
                    return grad

                param.register_hook(lambda grad: hook_fn(grad, x))

        # Train the model
        best_validation_loss = float('inf')
        epochs_no_improve = 0
        if self.optimizer_type == "NAdam":
            optimizer = torch.optim.NAdam(self.parameters.parameters(), lr=self.lr)
        elif self.optimizer_type == "RMSprop":
            optimizer = torch.optim.RMSprop(self.parameters.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.Adam(self.parameters.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # Track total loss for the epoch
            correct = 0  # Track correct predictions
            total = 0  # Track total samples

            for data, target in train_dataloader:
                # Forward pass
                predictions = self.net_predict(data)
                loss = F.cross_entropy(predictions, target)
                self.losses_1[-1].append(loss.item())
                reg_loss = self.regularization()
                loss += reg_loss
                self.losses_reg[-1].append(reg_loss.item())

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
                save_task_data(self.D_centroids.size(0), epoch, avg_valid_loss, self.parameters)
                if epoch % 20 == 0:
                    print(f"Validation Accuracy after Epoch [{epoch + 1}/{self.num_epochs}]: {valid_accuracy:.2f}%, "
                          f"Loss = {valid_loss / len(valid_dataloader):.4f},")

            if avg_valid_loss < best_validation_loss:
                # torch.save(self.parameters.state_dict(), "parameters.pth") # TODO
                best_validation_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.early_stop_patience:
                print(f"Validation Accuracy after Epoch [{epoch + 1}/{self.num_epochs}]: {valid_accuracy:.2f}%, "
                      f"Loss = {valid_loss / len(valid_dataloader):.4f},")
                print(f"Early stopping at epoch {epoch + 1}")
                # self.parameters.load_state_dict(torch.load("parameters.pth"))
                break
        self.save_original_parameters()

    def model_predict(self, distances):
        with torch.no_grad():
            return torch.argmax(self.net_predict(self.n_nearest_points(distances), normalize=False), -1)
