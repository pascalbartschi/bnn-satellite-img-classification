import abc
from collections import deque
import enum
import math
import pathlib
import typing
import warnings

import numpy as np
import torch
import torch.optim
import torch.utils.data
import tqdm
from matplotlib import pyplot as plt

from util import draw_reliability_diagram, cost_function, setup_seeds, calc_calibration_curve

# wsl --shutdown
# wsl --list --all
# wsl --unregister docker-desktop
# wsl --unregister docker-desktop-data
# C:\Windows\System32\dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
# C:\Windows\System32\dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
# cd C:\Users\paesc\OneDrive\docs\projects\probabilistic-artificial-intelligence-projects\2_BNN_satellite_img_classification
# docker build --tag task2 .;docker run --rm -v "%cd%:/results" task2

EXTENDED_EVALUATION = True
"""
Set `EXTENDED_EVALUATION` to `True` in order to generate additional plots on validation data.
"""

USE_PRETRAINED_MODEL = True
"""
If `USE_PRETRAINED_MODEL` is `True`, then MAP inference uses provided pretrained weights.
You should not modify MAP training or the CNN architecture before passing the hard baseline.
If you set the constant to `False` (to further experiment),
this solution always performs MAP inference before running your SWAG implementation.
Note that MAP inference can take a long time.
"""


def main():
    # raise RuntimeError(
    #     "This main() method is for illustrative purposes only"
    #     " and will NEVER be called when running your solution to generate your submission file!\n"
    #     "The checker always directly interacts with your SWAGInference class and evaluate method.\n"
    #     "You can remove this exception for local testing, but be aware that any changes to the main() method"
    #     " are ignored when generating your submission file."
    # )

    data_location = pathlib.Path.cwd() / "2_BNN_satellite_img_classification"
    model_location = pathlib.Path.cwd() / "2_BNN_satellite_img_classification"
    output_location = pathlib.Path.cwd() / "2_BNN_satellite_img_classification"

    # Load training data
    training_images = torch.from_numpy(np.load(data_location / "train_xs.npz")["train_xs"])
    training_metadata = np.load(data_location / "train_ys.npz")
    training_labels = torch.from_numpy(training_metadata["train_ys"])
    training_snow_labels = torch.from_numpy(training_metadata["train_is_snow"])
    training_cloud_labels = torch.from_numpy(training_metadata["train_is_cloud"])
    training_dataset = torch.utils.data.TensorDataset(training_images, training_snow_labels, training_cloud_labels, training_labels)

    # Load validation data
    validation_images = torch.from_numpy(np.load(data_location / "val_xs.npz")["val_xs"])
    validation_metadata = np.load(data_location / "val_ys.npz")
    validation_labels = torch.from_numpy(validation_metadata["val_ys"])
    validation_snow_labels = torch.from_numpy(validation_metadata["val_is_snow"])
    validation_cloud_labels = torch.from_numpy(validation_metadata["val_is_cloud"])
    validation_dataset = torch.utils.data.TensorDataset(validation_images, validation_snow_labels, validation_cloud_labels, validation_labels)

    # Fix all randomness
    setup_seeds()

    # Build and run the actual solution
    training_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    # swag_inference = SWAGInference(
    #     train_xs=training_dataset.tensors[0],
    #     model_dir=model_location,
    #     inference_mode=InferenceType.SWAG_FULL,
    #     swag_training_epochs=30,
    #     swag_lr=0.045,
    #     swag_update_interval=1,
    #     max_rank_deviation_matrix=15,
    #     num_bma_samples=30
    # )

    swag_inference = SWAGInference(
        train_xs=training_dataset.tensors[0],
        model_dir=model_location,
        inference_mode=InferenceType.SWAG_FULL,
        swag_training_epochs=1,
        swag_lr=0.035,
        swag_update_interval=1,
        max_rank_deviation_matrix=15,
        num_bma_samples=1
    )
    swag_inference.fit(training_loader)
    swag_inference.apply_calibration(validation_dataset)

    # fork_rng ensures that the evaluation does not change the rng state.
    # That way, you should get exactly the same results even if you remove evaluation
    # to save computational time when developing the task
    # (as long as you ONLY use torch randomness, and not e.g. random or numpy.random).
    with torch.random.fork_rng():
        evaluate(swag_inference, validation_dataset, EXTENDED_EVALUATION, output_location)


class InferenceType(enum.Enum):
    """
    Inference mode switch for your implementation.
    `MAP` simply predicts the most likely class using pretrained MAP weights.
    `SWAG_DIAGONAL` and `SWAG_FULL` correspond to SWAG-diagonal and the full SWAG method, respectively.
    """
    MAP = 0
    SWAG_DIAGONAL = 1
    SWAG_FULL = 2


class SWAGInference(object):
    """
    Your implementation of SWA-Gaussian.
    This class is used to run and evaluate your solution.
    You must preserve all methods and signatures of this class.
    However, you can add new methods if you want.

    We provide basic functionality and some helper methods.
    You can pass all baselines by only modifying methods marked with TODO.
    However, we encourage you to skim other methods in order to gain a better understanding of SWAG.
    """

    def __init__(
        self,
        train_xs: torch.Tensor,
        model_dir: pathlib.Path,
        # TODO(1): change inference_mode to InferenceMode.SWAG_DIAGONAL
        # TODO(2): change inference_mode to InferenceMode.SWAG_FULL
        inference_mode: InferenceType = InferenceType.SWAG_FULL,
        # TODO(2): optionally add/tweak hyperparameters
        swag_training_epochs: int = 30,
        swag_lr: float = 0.045,
        swag_update_interval: int = 1,
        max_rank_deviation_matrix: int = 15,
        num_bma_samples: int = 30,
    ):
        """
        :param train_xs: Training images (for storage only)
        :param model_dir: Path to directory containing pretrained MAP weights
        :param inference_mode: Control which inference mode (MAP, SWAG-diagonal, full SWAG) to use
        :param swag_training_epochs: Total number of gradient descent epochs for SWAG
        :param swag_lr: Learning rate for SWAG gradient descent
        :param swag_update_interval: Frequency (in epochs) for updating SWAG statistics during gradient descent
        :param max_rank_deviation_matrix: Rank of deviation matrix for full SWAG
        :param num_bma_samples: Number of networks to sample for Bayesian model averaging during prediction
        """

        self.model_dir = model_dir
        self.inference_mode = inference_mode
        self.swag_training_epochs = swag_training_epochs
        self.swag_lr = swag_lr
        self.swag_update_interval = swag_update_interval
        self.max_rank_deviation_matrix = max_rank_deviation_matrix
        self.num_bma_samples = num_bma_samples

        # Network used to perform SWAG.
        # Note that all operations in this class modify this network IN-PLACE!
        self.network = CNN(in_channels=3, out_classes=6)

        # Store training dataset to recalculate batch normalization statistics during SWAG inference
        self.training_dataset = torch.utils.data.TensorDataset(train_xs)

        # SWAG-diagonal attribute initialization a dictionary that maps from weight name to values
        self.theta_squared_bar = self._create_weight_copy()
        self.theta_SWA = self._create_weight_copy()

        # Full SWAG low rank matrix initialization
        self.D_hat = {layer: deque(maxlen=self.max_rank_deviation_matrix) for layer, _ in self.network.named_parameters()}

        # Calibration, prediction, and other attributes
        # self.temperature_scaling = TemperatureScaling()
        # confidence 5% above mean: 0.55
        self._adaptive_threshold_percentile = 0.55
        self.complex_calibration = False

        # scheduler tweaking
        #  this is the final learning rate at which the rate scheduler will end up
        self._swag_final_lr = 0.03
        #  define the decay type of the learning rate => best performance using linear
        self.lr_decay_type = "const" # "exponential" "cyclical" "const_linear"

    def update_swag_statistics(self) -> None:
        """
        Update SWAG statistics with the current weights of self.network.
        """

        # Create a copy of the current network weights
        copied_params = {name: param.detach() for name, param in self.network.named_parameters()}

        
        for layer, theta_i in copied_params.items():
            # update the running averages of the tracked SWAG attributes
            self.theta_SWA[layer] = ((self.T-self.swag_update_interval)/self.T) * self.theta_SWA[layer] + (1/self.T) * theta_i
            self.theta_squared_bar[layer] = ((self.T-self.swag_update_interval)/self.T) * self.theta_squared_bar[layer] + (1/self.T) * theta_i**2


        # Full SWAG
        if self.inference_mode == InferenceType.SWAG_FULL:
            for layer, theta_i in copied_params.items():
                self.D_hat[layer].appendleft(theta_i - self.theta_SWA[layer])
            # TODO(2): update full SWAG attributes for weight `name` using `copied_params` and `param`
            # raise NotImplementedError("Update full SWAG statistics")

    def fit_swag_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Fit SWAG on top of the pretrained network self.network.
        This method should perform gradient descent with occasional SWAG updates
        by calling self.update_swag_statistics().
        """

        # We use SGD with momentum and weight decay to perform SWA.
        # See the paper on how weight decay corresponds to a type of prior.
        # Feel free to play around with optimization hyperparameters.
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=self.swag_lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )

        # TODO(2): Update SWAGScheduler instantiation if you decided to implement a custom schedule.
        #  By default, this scheduler just keeps the initial learning rate given to `optimizer`.
        lr_scheduler = SWAGScheduler(
            optimizer,
            epochs=self.swag_training_epochs,
            steps_per_epoch=len(loader),
            final_lr=self._swag_final_lr,
            decay_type=self.lr_decay_type,
            decay_steps=1,
            cycle_length=30,
            start_epoch_decay=10
        )

        # TODO(1): Perform initialization for SWAG fitting
        # raise NotImplementedError("Initialize SWAG fitting")

        self.network.train()
        with tqdm.trange(self.swag_training_epochs, desc="Running gradient descent for SWA") as pbar:
            progress_dict = {}
            for epoch in pbar:
                self.T = epoch + 1
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                for batch_images, batch_snow_labels, batch_cloud_labels, batch_labels in loader:
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)
                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

                # TODO(1): Implement periodic SWAG updates using the attributes defined in __init__
                # running average of theta_SWA
                if self.T % self.swag_update_interval == 0: 
                    self.update_swag_statistics()
                

    def apply_calibration(self, validation_data: torch.utils.data.Dataset) -> None:
        """
        Calibrate your predictions using a small validation set.
        validation_data contains well-defined and ambiguous samples,
        where you can identify the latter by having label -1.
        """
        if self.inference_mode == InferenceType.MAP:
            # In MAP mode, simply predict argmax and do nothing else
            self._calibration_threshold = 0.0
            return

        # TODO(1): pick a prediction threshold, either constant or adaptive.
        #  The provided value should suffice to pass the easy baseline.
        # raise NotImplementedError("Learn about and implement adaptive thresholds!")
        self._calibration_threshold = 2.0 / 3.0

        if self.complex_calibration:
            # TODO(2): perform additional calibration if desired.
            #  Feel free to remove or change the prediction threshold.
            val_images, val_snow_labels, val_cloud_labels, val_labels = validation_data.tensors
            assert val_images.size() == (140, 3, 60, 60)  # N x C x H x W
            assert val_labels.size() == (140,)
            assert val_snow_labels.size() == (140,)
            assert val_cloud_labels.size() == (140,)

            ambiguous_mask = val_labels == -1
            snowy_cloudy_mask = (val_snow_labels == 1) | (val_cloud_labels == 1)
            well_defined_mask = ~ambiguous_mask & ~snowy_cloudy_mask
            # all label masks -> what rules make sense?

            # forward pass through the network 
            val_probabilities = self.predict_probabilities(val_images)
            # calculate the confidence
            val_confidence, val_max_likelihood_labels = torch.max(val_probabilities, dim = 1)

            # calculate the upper percentile of confidences
            self._class_calibration_thresholds = [torch.quantile(val_confidence[val_labels == i], 
                                                                self._adaptive_threshold_percentile).item() for i in range (6)]

            print("Calibration Threshold: ", self._class_calibration_thresholds)
        # find optimal scaling temperature
        # self.temperature_scaling.set_temperature(val_logits, val_labels)
        # apply the optimal temperature
        # self.temperature_scaled_logits = self.temperature_scaling()


    def predict_probabilities_swag(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Perform Bayesian model averaging using your SWAG statistics and predict
        probabilities for all samples in the loader.
        Outputs should be a Nx6 tensor, where N is the number of samples in loader,
        and all rows of the output should sum to 1.
        That is, output row i column j should be your predicted p(y=j | x_i).
        """

        self.network.eval()

        # Perform Bayesian model averaging:
        # Instead of sampling self.num_bma_samples networks (using self.sample_parameters())
        # for each datapoint, you can save time by sampling self.num_bma_samples networks,
        # and perform inference with each network on all samples in loader.
        model_predictions = []
        for _ in tqdm.trange(self.num_bma_samples, desc="Performing Bayesian model averaging"):
            # TODO(1): Sample new parameters for self.network from the SWAG approximate posterior
            # raise NotImplementedError("Sample network parameters")
            self.sample_parameters()

            loader_predictions = []
            # TODO(1): Perform inference for all samples in `loader` using current model sample,
            #  and add the predictions to model_predictions
            with torch.no_grad():  # Disable gradient calculations
                for batch_images, *_ in loader:
                    # Move data to the same device as the model
                    batch_images = batch_images.to(next(self.network.parameters()).device)

                    # Get model predictions
                    predictions = self.network(batch_images)
                    
                    # Store predictions in a list
                    loader_predictions.append(predictions.cpu())  # Move to CPU if necessary


            model_predictions.append(torch.cat(loader_predictions, dim=0))
            # raise NotImplementedError("Perform inference using current model")

        assert len(model_predictions) == self.num_bma_samples
        assert all(
            isinstance(sample_predictions, torch.Tensor)
            and sample_predictions.dim() == 2  # N x C
            and sample_predictions.size(1) == 6
            for sample_predictions in model_predictions
        )
        model_probabilities = torch.softmax(torch.stack(model_predictions), dim = -1)
        # Shape: [num_bma_samples, num_data_points, num_classes]
        # TODO(1): Average predictions from different model samples into bma_probabilities
        # raise NotImplementedError("Aggregate predictions from model samples")
        bma_probabilities = torch.mean(model_probabilities, dim=0)
        # bma_prediction_var = torch.var(model_predictions, dim=0)

        assert bma_probabilities.dim() == 2 and bma_probabilities.size(1) == 6  # N x C
        assert torch.allclose(bma_probabilities.sum(dim=1), torch.ones_like(bma_probabilities.sum(dim=1)), atol=1e-6), "Probabilities must sum to 1"
        return bma_probabilities

    def sample_parameters(self) -> None:
        """
        Sample a new network from the approximate SWAG posterior.
        For simplicity, this method directly modifies self.network in-place.
        Hence, after calling this method, self.network corresponds to a new posterior sample.
        """


        # Instead of acting on a full vector of parameters, all operations can be done on per-layer parameters.
        for layer, param in self.network.named_parameters():
            # SWAG-diagonal part:
            # We draw one sample for every parameter for the diagonal part of the covariance
            z1_diag = torch.randn(param.size())
            # TODO(1): Sample parameter values for SWAG-diagonal
            # raise NotImplementedError("Sample parameter for SWAG-diagonal")
            mean_weights = self.theta_SWA[layer]
            std_weights = (self.theta_squared_bar[layer] - self.theta_SWA[layer]**2)**0.5 # Sigma_diag -> this work component wise
            assert mean_weights.size() == param.size() and std_weights.size() == param.size()

            # Diagonal part
            sampled_weight = mean_weights + std_weights * z1_diag

            # Full SWAG part
            if self.inference_mode == InferenceType.SWAG_FULL:
                # TODO(2): Sample parameter values for full SWAG
                # raise NotImplementedError("Sample parameter for full SWAG")

                # Flatten each tensor in deque before stacking
                flattened_deviations = [dev.view(-1) for dev in self.D_hat[layer]]
                
                # Stack deviations to create a matrix of shape [param.numel(), num_snapshots]
                deviations_matrix = torch.stack(flattened_deviations, dim=1)
                
                # Create z2_full to match the second dimension of deviations_matrix
                # We draw one sample per snapshot to maintain the efficiency of the low rank approximation
                z2_full = torch.randn(deviations_matrix.size(1), device=deviations_matrix.device)
                
                # Calculate the full SWAG contribution and reshape to the original parameter shape
                full_swag_contribution = (deviations_matrix @ z2_full / ((2 * (self.max_rank_deviation_matrix - 1)) ** 0.5))
                sampled_weight += full_swag_contribution.view(param.size())  # Reshape to match param's shape

            # Modify weight value in-place; directly changing self.network
            param.data = sampled_weight

        # TODO(1): Don't forget to update batch normalization statistics using self._update_batchnorm_statistics()
        #  in the appropriate place!
        # raise NotImplementedError("Update batch normalization statistics for newly sampled network")
        self._update_batchnorm_statistics()

    def predict_labels(self, predicted_probabilities: torch.Tensor) -> torch.Tensor:
        """
        Predict labels in {0, 1, 2, 3, 4, 5} or "don't know" as -1
        based on your model's predicted probabilities.
        The parameter predicted_probabilities is an Nx6 tensor containing predicted probabilities
        as returned by predict_probabilities(...).
        The output should be a N-dimensional long tensor, containing values in {-1, 0, 1, 2, 3, 4, 5}.
        """
        # TODO check whether this result in effect that was intended
        # label_probabilities = self.temperature_scaling(predicted_probabilities)

        # label_probabilities contains the per-row maximum values in predicted_probabilities,
        # max_likelihood_labels the corresponding column index (equivalent to class).
        label_probabilities, max_likelihood_labels = torch.max(predicted_probabilities, dim=-1) 
        num_samples, num_classes = predicted_probabilities.size()
        assert label_probabilities.size() == (num_samples,) and max_likelihood_labels.size() == (num_samples,)

        # A model without uncertainty awareness might simply predict the most likely label per sample:

        # A bit better: use a threshold to decide whether to return a label or "don't know" (label -1)
        # TODO(2): implement a different decision rule if desired
        if self.complex_calibration: 
            # construct tensor to locally compare with label probabitie
            class_thresholds = torch.tensor([self._class_calibration_thresholds[label] for label in max_likelihood_labels])
            # update label probabilities based on temperature scaling
            return torch.where(
                label_probabilities >= class_thresholds,
                max_likelihood_labels,
                torch.ones_like(max_likelihood_labels) * -1,
            )
        else: 
            return torch.where(
                label_probabilities >= self._calibration_threshold,
                max_likelihood_labels,
                torch.ones_like(max_likelihood_labels) * -1,
            )
            

    def _create_weight_copy(self) -> typing.Dict[str, torch.Tensor]:
        """Create an all-zero copy of the network weights as a dictionary that maps name -> weight"""
        return {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.network.named_parameters()
        }

    def fit(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> None:
        """
        Perform full SWAG fitting procedure.
        If `PRETRAINED_WEIGHTS_FILE` is `True`, this method skips the MAP inference part,
        and uses pretrained weights instead.

        Note that MAP inference can take a very long time.
        You should hence only perform MAP inference yourself after passing the hard baseline
        using the given CNN architecture and pretrained weights.
        """

        # MAP inference to obtain initial weights
        PRETRAINED_WEIGHTS_FILE = self.model_dir / "map_weights.pt"
        if USE_PRETRAINED_MODEL:
            self.network.load_state_dict(torch.load(PRETRAINED_WEIGHTS_FILE))
            print("Loaded pretrained MAP weights from", PRETRAINED_WEIGHTS_FILE)
        else:
            self.fit_map_model(loader)

        # SWAG
        if self.inference_mode in (InferenceType.SWAG_DIAGONAL, InferenceType.SWAG_FULL):
            self.fit_swag_model(loader)

    def fit_map_model(self, loader: torch.utils.data.DataLoader) -> None:
        """
        MAP inference procedure to obtain initial weights of self.network.
        This is the exact procedure that was used to obtain the pretrained weights we provide.
        """
        map_training_epochs = 140
        initial_learning_rate = 0.01
        reduced_learning_rate = 0.0001
        start_decay_epoch = 50
        decay_factor = reduced_learning_rate / initial_learning_rate

        # Create optimizer, loss, and a learning rate scheduler that aids convergence
        optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=initial_learning_rate,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss(
            reduction="mean",
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [
                torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0),
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=decay_factor,
                    total_iters=(map_training_epochs - start_decay_epoch) * len(loader),
                ),
            ],
            milestones=[start_decay_epoch * len(loader)],
        )

        # Put network into training mode
        # Batch normalization layers are only updated if the network is in training mode,
        # and are replaced by a moving average if the network is in evaluation mode.
        self.network.train()
        with tqdm.trange(map_training_epochs, desc="Fitting initial MAP weights") as pbar:
            progress_dict = {}
            # Perform the specified number of MAP epochs
            for epoch in pbar:
                avg_loss = 0.0
                avg_accuracy = 0.0
                num_samples = 0
                # Iterate over batches of randomly shuffled training data
                for batch_images, _, _, batch_labels in loader:
                    # Training step
                    optimizer.zero_grad()
                    predictions = self.network(batch_images)
                    batch_loss = loss_fn(input=predictions, target=batch_labels)
                    batch_loss.backward()
                    optimizer.step()

                    # Save learning rate that was used for step, and calculate new one
                    progress_dict["lr"] = lr_scheduler.get_last_lr()[0]
                    with warnings.catch_warnings():
                        # Suppress annoying warning (that we cannot control) inside PyTorch
                        warnings.simplefilter("ignore")
                        lr_scheduler.step()

                    # Calculate cumulative average training loss and accuracy
                    avg_loss = (batch_images.size(0) * batch_loss.item() + num_samples * avg_loss) / (
                        num_samples + batch_images.size(0)
                    )
                    avg_accuracy = (
                        torch.sum(predictions.argmax(dim=-1) == batch_labels).item()
                        + num_samples * avg_accuracy
                    ) / (num_samples + batch_images.size(0))
                    num_samples += batch_images.size(0)

                    progress_dict["avg. epoch loss"] = avg_loss
                    progress_dict["avg. epoch accuracy"] = avg_accuracy
                    pbar.set_postfix(progress_dict)

    def predict_probabilities(self, xs: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities for the given images xs.
        This method returns an NxC float tensor,
        where row i column j corresponds to the probability that y_i is class j.

        This method uses different strategies depending on self.inference_mode.
        """
        self.network = self.network.eval()

        # Create a loader that we can deterministically iterate many times if necessary
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(xs),
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        with torch.no_grad():  # save memory by not tracking gradients
            if self.inference_mode == InferenceType.MAP:
                return self.predict_probabilities_map(loader)
            else:
                return self.predict_probabilities_swag(loader)

    def predict_probabilities_map(self, loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Predict probabilities assuming that self.network is a MAP estimate.
        This simply performs a forward pass for every batch in `loader`,
        concatenates all results, and applies a row-wise softmax.
        """
        all_predictions = []
        for (batch_images,) in loader:
            all_predictions.append(self.network(batch_images))

        all_predictions = torch.cat(all_predictions)
        return torch.softmax(all_predictions, dim=-1)

    def _update_batchnorm_statistics(self) -> None:
        """
        Reset and fit batch normalization statistics using the training dataset self.training_dataset.
        We provide this method for you for convenience.
        See the SWAG paper for why this is required.

        Batch normalization usually uses an exponential moving average, controlled by the `momentum` parameter.
        However, we are not training but want the statistics for the full training dataset.
        Hence, setting `momentum` to `None` tracks a cumulative average instead.
        The following code stores original `momentum` values, sets all to `None`,
        and restores the previous hyperparameters after updating batchnorm statistics.
        """

        original_momentum_values = dict()
        for module in self.network.modules():
            # Only need to handle batchnorm modules
            if not isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                continue

            # Store old momentum value before removing it
            original_momentum_values[module] = module.momentum
            module.momentum = None

            # Reset batch normalization statistics
            module.reset_running_stats()

        loader = torch.utils.data.DataLoader(
            self.training_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        self.network.train()
        for (batch_images,) in loader:
            self.network(batch_images)
        self.network.eval()

        # Restore old `momentum` hyperparameter values
        for module, momentum in original_momentum_values.items():
            module.momentum = momentum


class SWAGScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Custom learning rate scheduler that calculates a different learning rate each gradient descent step.
    The default implementation keeps the original learning rate constant, i.e., does nothing.
    You can implement a custom schedule inside calculate_lr,
    and add+store additional attributes in __init__.
    You should not change any other parts of this class.
    """
    # TODO(2): Add and store additional arguments if you decide to implement a custom scheduler✅
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        steps_per_epoch: int,
        final_lr: float,
        decay_type: str,
        cycle_length: int,
        decay_steps: int,
        start_epoch_decay: int
    ):
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.decay_type = decay_type
        self.final_lr = final_lr
        self.initial_lr = optimizer.param_groups[0]["lr"]
        self.decay_rate = optimizer.param_groups[0]["weight_decay"]
        self.decay_steps = decay_steps
        self.cycle_length = cycle_length
        self.start_epoch_decay = start_epoch_decay

        super().__init__(optimizer, last_epoch=-1, verbose=False)

    def calculate_lr(self, current_epoch: float, old_lr: float) -> float:
        """
        Calculate the learning rate for the epoch given by current_epoch.
        current_epoch is the fractional epoch of SWA fitting, starting at 0.
        That is, an integer value x indicates the start of epoch (x+1),
        and non-integer values x.y correspond to steps in between epochs (x+1) and (x+2).
        old_lr is the previous learning rate.

        This method should return a single float: the new learning rate.
        """
        # TODO(2): Implement a custom schedule if desired✅
        if self.decay_type == "linear":
            new_lr = self.initial_lr - (self.initial_lr - self.final_lr)*(current_epoch / self.epochs)
        elif self.decay_type == "exponential":
            new_lr = self.initial_lr * (self.decay_rate ** (current_epoch / self.decay_steps))
        elif self.decay_type == "cyclical":
            # it essentially decreases the learning rate linearly from alpha_1 to alpha_2
            # it is the same as mentioned in the paper: Averaging Weights Leads to Wider Optima and Better Generalization
            t = 1/self.cycle_length * (np.mod(current_epoch, self.cycle_length) + 1)
            new_lr = (1 - t)*self.initial_lr + t * self.final_lr
            pass
        elif self.decay_type == "const_linear":
            if current_epoch < self.start_epoch_decay:
                new_lr = old_lr
            else:
                new_lr = self.initial_lr - (self.initial_lr - self.final_lr)*((current_epoch - self.start_epoch_decay) / (self.epochs - self.start_epoch_decay))
        else:
            new_lr = old_lr
        return new_lr


    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )
        return [
            self.calculate_lr(self.last_epoch / self.steps_per_epoch, group["lr"])
            for group in self.optimizer.param_groups
        ]


def evaluate(
    swag_inference: SWAGInference,
    eval_dataset: torch.utils.data.Dataset,
    extended_evaluation: bool,
    output_location: pathlib.Path,
) -> None:
    """
    Evaluate your model.
    Feel free to change or extend this code.
    :param swag_inference: Trained model to evaluate
    :param eval_dataset: Validation dataset
    :param: extended_evaluation: If True, generates additional plots
    :param output_location: Directory into which extended evaluation plots are saved
    """

    print("Evaluating model on validation data")

    # We ignore is_snow and is_cloud here, but feel free to use them as well
    images, snow_labels, cloud_labels, labels = eval_dataset.tensors

    # Predict class probabilities on test data,
    # most likely classes (according to the max predicted probability),
    # and classes as predicted by your SWAG implementation.
    all_pred_probabilities = swag_inference.predict_probabilities(images)
    max_pred_probabilities, argmax_pred_labels = torch.max(all_pred_probabilities, dim=-1)
    predicted_labels = swag_inference.predict_labels(all_pred_probabilities)

    # Create a mask that ignores ambiguous samples (those with class -1)
    non_ambiguous_mask = labels != -1

    # Calculate three kinds of accuracy:
    # 1. Overall accuracy, counting "don't know" (-1) as its own class
    # 2. Accuracy on all samples that have a known label. Predicting -1 on those counts as wrong here.
    # 3. Accuracy on all samples that have a known label w.r.t. the class with the highest predicted probability.
    overall_accuracy = torch.mean((predicted_labels == labels).float()).item()
    non_ambiguous_accuracy = torch.mean((predicted_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()).item()
    non_ambiguous_argmax_accuracy = torch.mean(
        (argmax_pred_labels[non_ambiguous_mask] == labels[non_ambiguous_mask]).float()
    ).item()
    print(f"Accuracy (raw): {overall_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, your predictions): {non_ambiguous_accuracy:.4f}")
    print(f"Accuracy (non-ambiguous only, predicting most-likely class): {non_ambiguous_argmax_accuracy:.4f}")

    # Determine which threshold would yield the smallest cost on the validation data
    # Note that this threshold does not necessarily generalize to the test set!
    # However, it can help you judge your method's calibration.
    threshold_values = [0.0] + list(torch.unique(max_pred_probabilities, sorted=True))
    costs = []
    for threshold in threshold_values:
        thresholded_predictions = torch.where(max_pred_probabilities <= threshold, -1 * torch.ones_like(predicted_labels), predicted_labels)
        costs.append(cost_function(thresholded_predictions, labels).item())
    best_threshold_index = np.argmin(costs)
    print(f"Best cost {costs[best_threshold_index]} at threshold {threshold_values[best_threshold_index]}")
    print("Note that this threshold does not necessarily generalize to the test set!")

    # Calculate ECE and plot the calibration curve
    calibration_data = calc_calibration_curve(all_pred_probabilities.numpy(), labels.numpy(), num_bins=20)
    print("Validation ECE:", calibration_data["ece"])

    if extended_evaluation:
        identifier = f"T{swag_inference.swag_training_epochs}_LR{swag_inference.swag_lr}_UI{swag_inference.swag_update_interval}_MRDM{swag_inference.max_rank_deviation_matrix}_NBS{swag_inference.num_bma_samples}"
        print("Plotting reliability diagram")
        fig = draw_reliability_diagram(calibration_data)
        ax = fig.gca()  # Get the current axis
        ax.text(0, 0.8, f"ECE: {calibration_data['ece']:.6f}", ha='left', va='center', transform=ax.transAxes)
        ax.text(0, 0.7, f"Acc. {overall_accuracy:.4f}", ha='left', va='center', transform=ax.transAxes)
        ax.text(0, 0.6, f"ML Acc {non_ambiguous_argmax_accuracy:.4f}", ha='left', va='center', transform=ax.transAxes)
        fig.savefig(output_location / f"extended_eval/reliability_diagram_{identifier}.pdf")

        sorted_confidence_indices = torch.argsort(max_pred_probabilities)

        # Plot samples your model is most confident about
        print("Plotting most confident validation set predictions")
        most_confident_indices = sorted_confidence_indices[-10:]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = most_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Most confident predictions", size=20)
        fig.savefig(output_location / f"extended_eval/examples_most_confident_{identifier}.pdf")

        # Plot samples your model is least confident about
        print("Plotting least confident validation set predictions")
        least_confident_indices = sorted_confidence_indices[:10]
        fig, ax = plt.subplots(4, 5, figsize=(13, 11))
        for row in range(0, 4, 2):
            for col in range(5):
                sample_index = least_confident_indices[5 * row // 2 + col]
                ax[row, col].imshow(images[sample_index].permute(1, 2, 0).numpy())
                ax[row, col].set_axis_off()
                ax[row + 1, col].set_title(f"pred. {predicted_labels[sample_index]}, true {labels[sample_index]}")
                bar_colors = ["C0"] * 6
                if labels[sample_index] >= 0:
                    bar_colors[labels[sample_index]] = "C1"
                ax[row + 1, col].bar(
                    np.arange(6), all_pred_probabilities[sample_index].numpy(), tick_label=np.arange(6), color=bar_colors
                )
        fig.suptitle("Least confident predictions", size=20)
        fig.savefig(output_location / f"extended_eval/examples_least_confident_{identifier}.pdf")


class CNN(torch.nn.Module):
    """
    Small convolutional neural network used in this task.
    You should not modify this class before passing the hard baseline.

    Note that if you change the architecture of this network,
    you need to re-run MAP inference and cannot use the provided pretrained weights anymore.
    Hence, you need to set `USE_PRETRAINED_INIT = False` at the top of this file.
    """
    def __init__(
        self,
        in_channels: int,
        out_classes: int,
    ):
        super().__init__()

        self.layer0 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=5),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        self.pool1 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )
        self.pool2 = torch.nn.MaxPool2d((2, 2), stride=(2, 2))

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3),
        )

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.linear = torch.nn.Linear(64, out_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool2(x)
        x = self.layer5(x)

        # Average features over both spatial dimensions, and remove the now superfluous dimensions
        x = self.global_pool(x).squeeze(-1).squeeze(-1)

        logits = self.linear(x)

        return logits

import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        # Initialize the temperature parameter T, which is learnable
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)  # Start at T=1.0

    def forward(self, logits):
        # Scale logits by dividing by the temperature parameter T
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        """
        Tune the temperature parameter on the validation set to minimize NLL.
        
        Args:
            logits (torch.Tensor): The BMA logits of shape (n_samples, n_classes).
            labels (torch.Tensor): The true labels of shape (n_samples,).
        """
        self.eval()  # Set module to evaluation mode
        nll_criterion = nn.CrossEntropyLoss()  # Use NLL as the calibration objective
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)  # Optimizer for single parameter

        def eval_nll():
            # Apply temperature scaling and compute NLL
            scaled_logits = self.forward(logits)
            loss = nll_criterion(scaled_logits, labels)
            loss.backward()
            return loss

        # Optimize temperature to minimize NLL on validation set
        optimizer.step(eval_nll)
        # print(f'Optimal temperature: {self.temperature.item()}')


if __name__ == "__main__":
    main()
