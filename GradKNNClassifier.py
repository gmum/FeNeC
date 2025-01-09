import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split

from Classifier import Classifier


class GradKNNClassifier(Classifier):
    def __init__(self, optimizer="Adam", n_points=10, mode=0, num_epochs=100, lr=1e-3, early_stop_patience=10,
                 reg_type=None, reg_lambda=None, normalization_type=None, tanh_x=None, centroids_new_old_ratio=None,
                 dataloader_batch_size=64, study_name="GradKNNClassifier", verbose=True, *args, **kwargs):
        """
        Initializes the GradKNNClassifier.

        Parameters:
         - optimizer (string): Name of the optimizer to use.
         - n_points (int): Number of samples retained per class for future tasks.
         - mode (int): Classifier mode:
            * 0: Train common parameters for all classes.
            * 1: Train separate parameters for each class.
         - num_epochs (int): Number of training epochs.
         - lr (float): Learning rate for optimization.
         - early_stop_patience (int): Early stopping patience threshold.
         - reg_type (int): Regularization type (0 for none, 1 for L1, otherwise L2).
         - reg_lambda (float): Regularization weight.
         - tanh_x (float): Scaling factor for `tanh` activation (If 'None', then don't apply tanh).
         - normalization_type (int): Normalization type:
            * None: Don't use normalization.
            * 1: Use standardization.
            * 2: Use normalization.
         - centroids_new_old_ratio (float): Ratio of new task samples to old task centroids in training
                                            (If 'None', then don't apply centroids).
         - dataloader_batch_size (int): Batch size for the DataLoader.
         - study_name (string): Name of the wandb study.
         - verbose (bool): If True, print training details.
        """
        super().__init__(*args, **kwargs)
        self.optimizer = optimizer
        self.n_points = n_points
        self.mode = mode
        self.num_epochs = num_epochs
        self.lr = lr
        self.early_stop_patience = early_stop_patience
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.tanh_x = tanh_x
        self.normalization_type = normalization_type
        self.centroids_new_old_ratio = centroids_new_old_ratio
        self.dataloader_batch_size = dataloader_batch_size
        self.study_name = study_name
        self.verbose = verbose

        self.task_boundaries = torch.tensor([0])  # Tracks class boundaries for normalization

        # Initialize parameters
        self.parameters = torch.nn.ParameterDict()
        if self.mode == 0:
            for parameter_name in ['alpha', 'a', 'b']:
                new_param = torch.randn(1, device=self.device)
                if parameter_name == 'alpha':
                    new_param = torch.abs(new_param)
                self.parameters.update({parameter_name: torch.nn.Parameter(new_param)})
        else:
            for parameter_name in ['alpha', 'a', 'b', 'r']:
                self.parameters.update({parameter_name: torch.nn.Parameter(torch.empty(0, device=self.device))})
        self.original_parameters = None

        with open("wandb_key.txt", "r") as key_file:
            api_key = key_file.read().strip()

        # Login to W&B using the key
        wandb.login(key=api_key)

        wandb.init(
            project=study_name,
            config={**locals(), **kwargs}
        )

    def model_predict(self, distances):
        """
        Method for predicting class labels based on distances to training samples.

        Parameters:
         - distances (Tensor): Distances of shape [batch_size, n_classes, n_centroids].

        Returns:
         - Tensor: Predicted class labels.
        """

        is_last_task = distances.size(1) == 100

        if is_last_task:
            wandb.finish()
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
            logits = parameters['alpha'] * data_sum
            return logits
        else:
            # Separate parameters for each class
            if self.tanh_x is not None:
                # Apply tanh to parameters before transforming the data
                data_transformed = (torch.tanh(parameters['a'])[None, :, None] * self.tanh_x +
                                    torch.tanh(parameters['b'])[None, :, None] * self.tanh_x * torch.log(data + 1e-16))
                data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
                data_sum = data_activated.sum(dim=-1)
                logits = (torch.tanh(parameters['alpha'][None, :] * self.tanh_x) * data_sum
                          + torch.tanh(parameters['r'][None, :]) * self.tanh_x)
            else:
                # Use raw parameters for transformation
                data_transformed = (parameters['a'][None, :, None] +
                                    parameters['b'][None, :, None] * torch.log(data + 1e-16))
                data_activated = F.leaky_relu(data_transformed, negative_slope=0.01)
                data_sum = data_activated.sum(dim=-1)
                logits = parameters['alpha'][None, :] * data_sum + parameters['r'][None, :]

            # Standardize logits during training if specified
            if is_training and self.normalization_type == 1:
                logits = torch.cat([(logits[:, start:end] - logits[:, start:end].mean()) / logits[:, start:end].std()
                                    for start, end in zip(self.task_boundaries[:-1], self.task_boundaries[1:])], dim=1)

            elif is_training and self.normalization_type == 2:
                logits = torch.cat([logits[:, start:end] / (1 + logits[:, start:end].std())
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

        # Train the model on the task data.
        self.train()

        # Save the current task parameters for regularization in future tasks.
        self.save_original_parameters()

    def train(self):
        """ Main training loop for the classifier. """
        # Prepare the DataLoaders for training and validation
        train_dataloader, train_dataloader_sec, valid_dataloader = self.prepare_dataloader()

        if train_dataloader_sec is not None:
            iter_dataloader_sec = iter(train_dataloader_sec)  # Iterator for additional centroids data

        # Initialize variables for early stopping and tracking metrics
        best_validation_loss = float('inf')
        epochs_no_improve = 0  # Count of consecutive epochs without improvement
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(self.parameters.parameters(), lr=self.lr)
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(self.parameters.parameters(), lr=self.lr)
        for epoch in range(self.num_epochs):
            epoch_loss = 0.0  # Cumulative loss for the epoch
            correct = 0  # Number of correct predictions
            total = 0  # Total number of samples

            for data, target in train_dataloader:
                # Concatenate training data with centroids data if provided
                if train_dataloader_sec is not None:
                    try:
                        data_sec, target_sec = next(iter_dataloader_sec)
                    except StopIteration:
                        # Reinitialize iterator if exhausted
                        iter_dataloader_sec = iter(train_dataloader_sec)
                        data_sec, target_sec = next(iter_dataloader_sec)
                    # Combine data and targets
                    data = torch.cat([data, data_sec])
                    target = torch.cat([target, target_sec])

                # Forward pass through the model
                predictions = self.gradknn_predict(data)
                loss = F.cross_entropy(predictions, target)  # Cross-entropy loss
                reg_loss = self.regularization()  # Regularization loss
                if self.verbose:
                    wandb.log({"cross_entropy_loss": loss, "reg_loss": reg_loss, "total_loss": loss + reg_loss})
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

                if self.verbose:
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
                    wandb.log({"valid_loss": avg_valid_loss, "valid_accuracy": 100.0 * valid_correct / valid_total})

            # Calculate and display epoch metrics (accuracy and loss)
            epoch_accuracy = 100.0 * correct / total  # TODO: use it, or remove it
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
            if not self.verbose:
                wandb.log({"cross_entropy_loss": epoch_loss / len(train_dataloader), "reg_loss": reg_loss,
                           "total_loss": loss + reg_loss})
                wandb.log({"valid_loss": avg_valid_loss, "valid_accuracy": 100.0 * valid_correct / valid_total})
            valid_accuracy = 100.0 * valid_correct / valid_total

            self.save_task_data(self.D_centroids.size(0), epoch, self.parameters)

            if self.verbose:
                # Save parameters for later study.
                if epoch % 10 == 0:
                    print(f"Validation Accuracy after Epoch [{epoch + 1}/{self.num_epochs}]: {valid_accuracy:.2f}%, "
                          f"Loss = {valid_loss / len(valid_dataloader):.4f},")

            # Early stopping logic: track and compare validation loss.
            if avg_valid_loss < best_validation_loss:
                # Save the best data for early stopping
                torch.save(self.parameters.state_dict(), f"{self.study_name}.pth")
                best_validation_loss = avg_valid_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Early stopping
            if epochs_no_improve >= self.early_stop_patience:
                # Stop training early if no improvement is seen for `early_stop_patience` epochs
                print(f"Early Stopping at Epoch [{epoch + 1}/{self.num_epochs}]: Accuracy = {valid_accuracy:.2f}%, "
                      f"Loss = {valid_loss / len(valid_dataloader):.4f},")
                # Load the best data
                self.parameters.load_state_dict(torch.load(f"{self.study_name}.pth", weights_only=True))
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
            self.D_centroids,
            d_class,
            self.batch_size)
            for d_class in self.D])
        # X shape: [points in the current task, n_classes or classes in the current task, self.n_points]

        # Construct corresponding labels.
        y = torch.arange(self.D_centroids.size(0) - self.D.size(0), self.D_centroids.size(0))
        y = y.repeat_interleave(self.D.size(1)).to(self.device)

        # Split the dataset into training and validation subsets (validation for early stopping)
        train_dataset, valid_dataset = self.train_test_split(TensorDataset(X, y), 0.9)
        train_dataloader_c = None  # define a train_dataloader_c, for a later easy check whether it's `None`

        # Create a separate dataloader with old centroids
        if self.centroids_new_old_ratio is not None:
            # Use all centroids, including the current ones, or only the old ones
            D_range = self.D_centroids.size(0)

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
                                                batch_size=int((1 - self.centroids_new_old_ratio)
                                                               * self.dataloader_batch_size))

        # Create DataLoaders for training and validation.
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.dataloader_batch_size, shuffle=False)

        data_loader_batch_size = self.dataloader_batch_size
        if train_dataloader_c is not None:
            data_loader_batch_size = int(self.centroids_new_old_ratio * data_loader_batch_size)
        train_dataloader = DataLoader(train_dataset, batch_size=data_loader_batch_size, shuffle=True, drop_last=True)

        if self.verbose:
            print("Dataloader created")

        # Return the dataloader with the smallest one being the first
        if train_dataloader_c is None or len(train_dataloader) < len(train_dataloader_c):
            return train_dataloader, train_dataloader_c, valid_dataloader
        else:
            return train_dataloader_c, train_dataloader, valid_dataloader

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
        reg_loss = 0  # Regularization loss
        if self.reg_type is not None and self.original_parameters is not None:
            for parameter in self.parameters:
                if self.mode == 0:
                    # Compute the difference between current parameters, and parameters from the previous task
                    diff = self.parameters[parameter] - self.original_parameters[parameter]
                else:
                    # Compute the difference between current and original parameters for each previous task's parameter
                    prev_num_classes = len(self.original_parameters[parameter])
                    diff = (self.parameters[parameter][:prev_num_classes] - self.original_parameters[parameter])

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
        if self.mode == 0 or self.original_parameters is None:
            # If the mode is 0, or the mode is 1, but it is the first task, deep copy the current parameters
            self.original_parameters = (
                torch.nn.ParameterDict({key: value.clone() for key, value in self.parameters.items()}))
        else:
            for parameter in self.parameters:
                # Append the newly added parameters for the current task to the original parameters
                self.original_parameters[parameter] = torch.nn.Parameter(torch.cat(
                    [self.original_parameters[parameter], self.parameters[parameter][-self.D.size(0):]], dim=0))

    def save_task_data(self, task, epoch, parameters):
        """
        Save task data to Weights & Biases (W&B).

        Parameters:
        - task (str): The name of the task.
        - epoch (int): The current epoch.
        - loss (float): Loss value for the epoch.
        - parameters (dict): Dictionary containing model parameters (`alpha`, `a`, `b`, `r`).
        """
        class_num = 100

        if self.mode == 0:
            data = {
                "task": task,
                "epoch": epoch,
                "alpha": parameters["alpha"].item(),
                "a": parameters["a"].item(),
                "b": parameters["b"].item(),
            }
            wandb.log(data)

        elif self.mode == 1:
            # Convert parameters to lists and pad to class_num with zeros
            def pad_to_all_classes(param):
                param_list = param.detach().cpu().numpy().tolist()
                return param_list + [0] * (class_num - len(param_list))

            alpha_list = pad_to_all_classes(parameters["alpha"])
            a_list = pad_to_all_classes(parameters["a"])
            b_list = pad_to_all_classes(parameters["b"])
            r_list = pad_to_all_classes(parameters["r"])

            # Prepare the data dictionary for W&B logging
            data = {
                "task": task,
                "epoch": epoch,
                **{f"alpha_{i}": alpha_list[i] for i in range(class_num)},
                **{f"a_{i}": a_list[i] for i in range(class_num)},
                **{f"b_{i}": b_list[i] for i in range(class_num)},
                **{f"r_{i}": r_list[i] for i in range(class_num)},
            }

            # Log data to W&B
            wandb.log(data)

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
