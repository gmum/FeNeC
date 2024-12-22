import csv
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split

from Classifier import Classifier


class GradKNNClassifier(Classifier):
    def __init__(self, n_points=10, mode=1, num_epochs=100, lr=1e-3, early_stop_patience=10, train_previous=True,
                 reg_type=1, reg_lambda=0.1, use_tanh=False, tanh_x=2, use_standardization=False, add_centroids=True,
                 only_prev_centroids=True, new_old_ratio=1, dataloader_batch_size=128, verbose=True, *args, **kwargs):
        """
        Initializes the GradKNNClassifier.

        Parameters:
         - n_points (int): Number of samples retained per class for future tasks.
         - mode (int): Classifier mode:
            * 0: Train common parameters for all classes.
            * 1: Train separate parameters for each class.
         - num_epochs (int): Number of training epochs.
         - lr (float): Learning rate for optimization.
         - early_stop_patience (int): Early stopping patience threshold.
         - train_previous (bool): If False, freezes parameters from previous tasks.
         - reg_type (int): Regularization type (0 for none, 1 for L1, otherwise L2).
         - reg_lambda (float): Regularization weight.
         - use_tanh (bool): Whether to apply `tanh` activation on logits.
         - tanh_x (float): Scaling factor for `tanh` activation.
         - use_standardization (bool): If True, standardize logits during training.
         - add_centroids (bool): If True, include centroids in training.
         - only_prev_centroids (bool): If True, exclude centroids of the current task.
         - new_old_ratio (float): Ratio of new task samples to old task centroids in training.
         - dataloader_batch_size (int): Batch size for the DataLoader.
         - verbose (bool): If True, print training details.
        """
        super().__init__(*args, **kwargs)
        self.n_points = n_points
        self.mode = mode
        self.num_epochs = num_epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.train_previous = train_previous
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.use_tanh = use_tanh
        self.tanh_x = tanh_x
        self.use_standardization = use_standardization
        self.add_prev_centroids = add_centroids
        self.only_prev_centroids = only_prev_centroids
        self.new_old_ratio = new_old_ratio
        self.dataloader_batch_size = dataloader_batch_size
        self.verbose = verbose

        self.task_boundaries = torch.tensor([0])  # Tracks class boundaries for normalization

        # Initialize parameters
        self.parameters = torch.nn.ParameterDict()
        if self.mode == 0:
            for parameter_name in ['alpha', 'a', 'b']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.randn(1, device=self.device))})
        else:
            for parameter_name in ['alpha', 'a', 'b', 'r']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.empty(0, device=self.device))})
            self.original_parameters = self.parameters.copy()  # for use in regularization

    def model_predict(self, distances):
        """
        Method for predicting class labels based on distances to training samples.

        Parameters:
         - distances (Tensor): Distances of shape [batch_size, n_classes, n_centroids].

        Returns:
         - Tensor: Predicted class labels.
        """
        with torch.no_grad():
            return torch.argmax(self.gradknn_predict(self.n_nearest_points(distances), is_training=False), -1)

    def gradknn_predict(self, data, is_training=True):
        """
        Compute predictions based on the current model parameters.

        Parameters:
         - data (torch.Tensor): Input data.
         - is_training (bool): If True, applies training-specific normalization.

        Returns:
         - torch.Tensor: Predicted logits.
        """
        parameters = self.parameters

        if self.mode == 0:
            # Shared parameters for all classes
            data_transformed = parameters['a'] + parameters['b'] * torch.log(data + 1e-16)
            data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
            data_sum = data_activated.sum(dim=-1)
            logits = F.softplus(parameters['alpha']) * data_sum
            return logits
        else:
            # Separate parameters for each class
            if self.use_tanh:
                # Apply tanh to parameters before transforming the data
                data_transformed = (torch.tanh(parameters['a'])[None, :, None] * self.tanh_x +
                                    torch.tanh(parameters['b'])[None, :, None] * self.tanh_x * torch.log(data + 1e-16))
                data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
                data_sum = data_activated.sum(dim=-1)
                logits = (F.softplus(torch.tanh(parameters['alpha'][None, :]) * self.tanh_x) * data_sum
                          + torch.tanh(parameters['r'][None, :]) * self.tanh_x)
            else:
                # Use raw parameters for transformation
                data_transformed = (parameters['a'][None, :, None] +
                                    parameters['b'][None, :, None] * torch.log(data + 1e-16))
                data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
                data_sum = data_activated.sum(dim=-1)
                logits = F.softplus(parameters['alpha'][None, :]) * data_sum + parameters['r'][None, :]

            # Standardize logits during training if specified
            if is_training and self.use_standardization is True:
                logits = torch.cat([(logits[:, start:end] - logits[:, start:end].mean()) / logits[:, start:end].std()
                                    for start, end in zip(self.task_boundaries[:-1], self.task_boundaries[1:])], dim=1)

            return logits

    def fit(self, D, **kwargs):
        """
        Trains the classifier on the current task data.

        Parameters:
         - D (torch.Tensor): Training data for the current task. Used solely for passing to the base classifier.
        """
        super().fit(D)  # `self.D` (current task data) and `self.D_centroids` (class centroids) are now accessible.

        # Update task boundaries to keep track of class indices for normalization purposes.
        self.task_boundaries = torch.cat([self.task_boundaries, torch.tensor(self.D_centroids.size(0)).unsqueeze(0)])

        # Create and initialize new parameters for the current task (or use previous ones in the case of mode 0).
        self.update_parameters()

        # Freeze gradients for earlier tasks if we do not want to train them.
        freeze_num = self.D_centroids.size(0) - self.D.size(0)  # Number of elements to freeze.
        if not self.train_previous:
            for name, param in self.parameters.items():
                def hook_fn(grad, num_freeze=freeze_num):
                    # Zero out the gradient for the first `num_freeze` elements
                    grad[:num_freeze] = 0
                    return grad

                # Register the gradient modification hook for this parameter
                param.register_hook(lambda grad: hook_fn(grad, freeze_num))

        # Train the model on the task data.
        self.train()

        # Save the current task parameters for regularization in future tasks.
        self.save_original_parameters()

    def train(self):
        """ Main training loop for the classifier. """
        # Prepare the DataLoaders for training and validation
        train_dataloader, train_dataloader_c, valid_dataloader = self.prepare_dataloader()

        # Initialize variables for early stopping and tracking metrics
        best_validation_loss = float('inf')
        epochs_no_improve = 0  # Count of consecutive epochs without improvement
        optimizer = torch.optim.Adam(self.parameters.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # Cumulative loss for the epoch
            correct = 0  # Number of correct predictions
            total = 0  # Total number of samples

            if train_dataloader_c is not None:
                iter_dataloader_c = iter(train_dataloader_c)  # Iterator for additional centroids data

            for data, target in train_dataloader:
                # Concatenate training data with centroids data if provided
                if train_dataloader_c is not None:
                    try:
                        data_c, target_c = next(iter_dataloader_c)
                    except StopIteration:
                        # Reinitialize iterator if exhausted
                        iter_dataloader_c = iter(train_dataloader_c)
                        data_c, target_c = next(iter_dataloader_c)
                    # Combine data and targets
                    data = torch.cat([data, data_c])
                    target = torch.cat([target, target_c])

                # Forward pass through the model
                predictions = self.gradknn_predict(data)
                loss = F.cross_entropy(predictions, target)  # Cross-entropy loss
                reg_loss = self.regularization()  # Regularization loss
                loss += reg_loss  # Add regularization to the total loss

                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Accumulate loss and calculate batch accuracy
                epoch_loss += loss.item()
                predicted_classes = torch.argmax(predictions, dim=1)
                correct += (predicted_classes == target).sum().item()
                total += target.size(0)

            # Calculate and display epoch metrics (accuracy and loss)
            epoch_accuracy = 100.0 * correct / total
            valid_correct = 0
            valid_total = 0
            valid_loss = 0.0
            with torch.no_grad():
                # Evaluate validation dataset
                for data, target in valid_dataloader:
                    predictions = self.gradknn_predict(data)
                    predicted_classes = torch.argmax(predictions, dim=1)
                    valid_correct += (predicted_classes == target).sum().item()
                    valid_total += target.size(0)
                    valid_loss += F.cross_entropy(predictions, target).item()
            avg_valid_loss = valid_loss / len(valid_dataloader)
            valid_accuracy = 100.0 * valid_correct / valid_total

            if self.verbose:
                # Save parameters for later study.
                self.save_task_data(self.D_centroids.size(0), epoch, avg_valid_loss, self.parameters)
                if epoch % 20 == 0:
                    print(f"Validation Accuracy after Epoch [{epoch + 1}/{self.num_epochs}]: {valid_accuracy:.2f}%, "
                          f"Loss = {valid_loss / len(valid_dataloader):.4f},")

            # Early stopping logic: track and compare validation loss.
            if avg_valid_loss < best_validation_loss:
                # Save the best data for early stopping (commented out for faster runs, as it shouldn't change much)
                # torch.save(self.parameters.state_dict(), "parameters.pth")
                best_validation_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.early_stop_patience:
                # Stop training early if no improvement is seen for `early_stop_patience` epochs
                print(f"Early Stopping at Epoch [{epoch + 1}/{self.num_epochs}]: Accuracy = {valid_accuracy:.2f}%, "
                      f"Loss = {valid_loss / len(valid_dataloader):.4f},")
                # Load the best data (commented out for faster runs, as it shouldn't change much)
                # self.parameters.load_state_dict(torch.load("parameters.pth"))
                break

    def prepare_dataloader(self):
        """
        Prepare DataLoaders for training and validation, including optional centroids.

        Returns:
         - train_dataloader (DataLoader): DataLoader for training.
         - train_dataloader_c (DataLoader): DataLoader for centroids (if applicable, `None` otherwise).
         - valid_dataloader (DataLoader): DataLoader for validation.
        """
        # Calculate features for all points in the current task
        X = torch.cat([self.metric.calculate_batch(
            self.n_nearest_points,
            # For mode 1, train all the classes, and for mode 0, train only the current task:
            self.D_centroids[(0 if self.mode == 1 else -self.D.size(0)):],
            d_class,
            self.batch_size)
            for d_class in self.D])
        # X shape: [points in the current task, n_classes / classes in the current task, self.n_points]

        # Construct corresponding labels.
        if self.mode == 0:
            y = torch.arange(self.D.size(0))
        else:
            y = torch.arange(self.D_centroids.size(0) - self.D.size(0), self.D_centroids.size(0))
        y = y.repeat_interleave(self.D.size(1)).to(self.device)

        # Split the dataset into training and validation subsets (validation for early stopping)
        train_dataset, valid_dataset = self.train_test_split(TensorDataset(X, y), 0.9)
        train_dataloader_c = None  # define a train_dataloader_c, for a later easy check whether it's `None`

        # Create a separate dataloader with old centroids
        if self.add_prev_centroids:
            # Use all centroids, including the current ones, or only the old ones
            D_range = self.D_centroids.size(0)
            if self.only_prev_centroids:
                D_range -= self.D.size(0)

            # Create a dataloader with centroids if required
            if D_range != 0:
                X_centroids = torch.cat([self.metric.calculate_batch(
                    # lambda function calculating the nearest centroids, excluding the same centroids:
                    lambda distances: self.n_nearest_points_centroids(distances, class_num),
                    self.D_centroids,
                    self.D_centroids[class_num],
                    self.batch_size)
                    for class_num in range(D_range)])

                y_centroids = (torch.arange(D_range).repeat_interleave(self.D_centroids.size(1)).to(self.device))

                train_dataset_c, valid_dataset_c = self.train_test_split(TensorDataset(X_centroids, y_centroids))
                valid_dataset = ConcatDataset([valid_dataset, valid_dataset_c])

                train_dataloader_c = DataLoader(train_dataset_c, shuffle=True, drop_last=True,
                                                batch_size=int((1 - self.new_old_ratio) * self.dataloader_batch_size))

        # Create DataLoaders for training and validation.
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.dataloader_batch_size, shuffle=False)

        data_loader_batch_size = self.dataloader_batch_size
        if train_dataloader_c is not None:
            data_loader_batch_size = int(self.new_old_ratio * data_loader_batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size=data_loader_batch_size, shuffle=True, drop_last=True)

        if self.verbose:
            print("Dataloader created")

        return train_dataloader, train_dataloader_c, valid_dataloader

    def update_parameters(self, init_range=(-1, 1)):
        """
        Update model parameters to accommodate new classes.

        Parameters:
         - new_classes (int): Number of new classes to add.
         - init_range (tuple): Range for initializing new parameters.
        """
        if self.mode == 1:  # Only applicable in mode 1 (per-class parameters)
            for parameter in self.parameters:
                # Generate new parameters uniformly in the given range
                new_param = ((init_range[1] - init_range[0]) * torch.rand(self.D.size(0), device=self.device)
                             + init_range[0])
                if parameter == 'alpha':
                    # Ensure 'alpha' remains positive by taking the absolute value
                    new_param = torch.abs(new_param)
                # Concatenate new parameters for the current task with existing ones
                self.parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.parameters[parameter], torch.nn.Parameter(new_param)], dim=0))

    def regularization(self):
        """
        Compute regularization loss based on differences between current and original parameters.

        Returns:
         - torch.Tensor: Regularization loss value.
        """
        reg_loss = 0  # Initialize regularization loss
        if self.mode == 1 and self.reg_type != 0:  # Only apply regularization in mode 1
            prev_num_classes = len(self.original_parameters['a'])
            for parameter in self.parameters:
                # Compute the difference between current and original parameters for each previous task's parameter
                diff = (self.parameters[parameter][:prev_num_classes]
                        - self.original_parameters[parameter])
                if self.reg_type == 1:
                    # Apply L1 regularization (sum of absolute differences)
                    reg_loss += self.reg_lambda * torch.sum(torch.abs(diff))
                else:
                    # Apply L2 regularization (sum of squared differences)
                    reg_loss += self.reg_lambda * torch.sum(diff ** 2)
        return reg_loss  # Return total regularization loss

    def n_nearest_points(self, distances):
        """
        Select the closest `n_points` samples for each class.

        Parameters:
         - distances (torch.Tensor): Pairwise distances with shape: [batch_size, n_classes, samples_per_class].

        Returns:
         - torch.Tensor: Distances to the closest `n_points` samples
                         with shape: [batch_size, self.n_classes, self.n_points].
        """
        return torch.topk(distances, self.n_points, sorted=True, largest=False)[0]

    def n_nearest_points_centroids(self, distances, class_num=-1):
        """
        Select the closest centroids for all classes, excluding one specific class. The selected class's samples are
        centroids, so it has a distance equal to 0 to one of the centroids. Therefore, we should exclude the
        closest centroid for this class.

        Parameters:
         - distances (torch.Tensor): Pairwise distances with shape: [batch_size, n_classes, samples_per_class].
         - class_num (int): Class index to exclude from selection.

        Returns:
         - torch.Tensor: Distances to the selected centroids.
        """
        # Retrieve the `n_points + 1` nearest centroids to account for exclusions
        nearest_points = torch.topk(distances, self.n_points + 1, sorted=True, largest=False)[0]
        # Exclude the closest centroid for the specified class and the furthest centroid for the other classes
        nearest_points = torch.cat([nearest_points[:, :class_num, :-1],
                                    nearest_points[:, class_num:(class_num + 1), 1:],
                                    nearest_points[:, (class_num + 1):, :-1]], dim=1)
        return nearest_points

    def save_original_parameters(self):
        """ Save original parameters for use in regularization. """
        if self.mode == 1:  # Only relevant in mode 1
            for parameter in self.parameters:
                # Append the newly added parameters for the current task to the original parameters
                self.original_parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.original_parameters[parameter], self.parameters[parameter][-self.D.size(0):]], dim=0))

    @staticmethod
    def train_test_split(dataset, train_ratio=0.9):
        """
        Split the dataset into training and validation subsets.

        Parameters:
         - dataset (torch.utils.data.Dataset): The dataset to split.
         - train_ratio (float): Proportion of the dataset to use for training.

        Returns:
         - tuple: (train_dataset, valid_dataset) subsets.
        """
        # Calculate the sizes for training and validation datasets
        train_size = int(train_ratio * len(dataset))
        valid_size = len(dataset) - train_size

        # Divide the dataset
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        return train_dataset, valid_dataset

    @staticmethod
    def save_task_data(task, epoch, loss, parameters, filename="data.csv"):
        """
        Save task data to a CSV file. Appends to an existing file or creates a new one.

        Parameters:
         - task (str): The name of the task.
         - epoch (int): The current epoch.
         - loss (float): Loss value for the epoch.
         - parameters (dict): Dictionary containing model parameters (`alpha`, `a`, `b`, `r`).
         - filename (str): The name of the CSV file to save data to (default is "data.csv").
        """
        class_num = 100

        # Convert parameters to lists and pad to all_classes with zeros
        def pad_to_all_classes(param):
            param_list = param.detach().cpu().numpy().tolist()
            return param_list + [0] * (class_num - len(param_list))

        alpha_list = pad_to_all_classes(parameters["alpha"])
        a_list = pad_to_all_classes(parameters["a"])
        b_list = pad_to_all_classes(parameters["b"])
        r_list = pad_to_all_classes(parameters["r"])

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
                        [f"alpha_{i}" for i in range(class_num)] +
                        [f"a_{i}" for i in range(class_num)] +
                        [f"b_{i}" for i in range(class_num)] +
                        [f"r_{i}" for i in range(class_num)]
                )
                writer.writerow(header)

            # Write the data row
            writer.writerow(row)
