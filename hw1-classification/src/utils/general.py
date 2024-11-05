from typing import List, Tuple, Union

import torch
import numpy as np


def reshape_to_vectors(*arrays: np.ndarray) -> List[np.ndarray]:
    """Reshape the input arrays to 2D vectors, preserving the first dimension.

    Args:
        *arrays (np.ndarray): One or more NumPy arrays to be reshaped.

    Returns:
        List[np.ndarray]: A list of reshaped 2D arrays, where each array is
        flattened except for the first dimension.
    """
    return [array.reshape(array.shape[0], -1) for array in arrays]


def normalize(*arrays: np.ndarray) -> List[np.ndarray]:
    """Normalize the input arrays using the mean and standard deviation.

    Args:
        *arrays (np.ndarray): One or more NumPy arrays to be normalized.

    Returns:
        List[np.ndarray]: A list of normalized arrays with mean 0 and standard deviation 1.

    Note:
        All arrays are normalized using a common mean and standard deviation computed
        from concatenating the input arrays along the first axis.
    """
    # Calculate the common mean and standard deviation
    mean = np.mean(np.concatenate(arrays), axis=0)
    std = np.std(np.concatenate(arrays), axis=0)
    return [(array - mean) / std for array in arrays]


def normalize_torch(*arrays: torch.Tensor) -> List[torch.Tensor]:
    """Normalize the input arrays using the mean and standard deviation.

    Args:
        *arrays (torch.Tensor): One or more PyTorch tensors to be normalized.

    Returns:
        List[torch.Tensor]: A list of normalized tensors with mean 0 and standard deviation 1.

    Note:
        All tensors are normalized using a common mean and standard deviation computed
        from concatenating the input tensors along the first axis.
    """
    print(f"arrays: {arrays[0].shape}")
    # Calculate the common mean and standard deviation
    mean = torch.mean(torch.cat(arrays, dim=0), dim=0).float()
    std = torch.std(torch.cat(arrays, dim=0), dim=0).float()
    return {
        "normalized_arrays": [(array - mean) / std for array in arrays],
        "mean": mean,
        "std": std,
    }


def dataset_stats(
    X_train: Union[np.ndarray, torch.Tensor],
    y_train: Union[np.ndarray, torch.Tensor],
    X_val: Union[np.ndarray, torch.Tensor],
    y_val: Union[np.ndarray, torch.Tensor],
    X_test: Union[np.ndarray, torch.Tensor],
    y_test: Union[np.ndarray, torch.Tensor],
    verbose: bool = False,
) -> Tuple[int, int, int]:
    """Compute and optionally print statistics of the dataset.

    Args:
        X_train (np.ndarray): Training data.
        y_train (np.ndarray): Training labels.
        X_val (np.ndarray): Validation data.
        y_val (np.ndarray): Validation labels.
        X_test (np.ndarray): Test data.
        y_test (np.ndarray): Test labels.
        verbose (bool, optional): If True, prints detailed statistics. Default is False.

    Returns:
        Tuple[int, int, int]: A tuple containing:
            - The number of features per sample.
            - The number of unique classes.
            - The total number of samples across training, validation, and test datasets.
    """
    num_features = (
        X_train[0].size if isinstance(X_train, np.ndarray) else X_train[0].numel()
    )
    num_classes = (
        len(np.unique(y_train))
        if isinstance(y_train, np.ndarray)
        else len(torch.unique(y_train))
    )
    num_samples = len(X_train) + len(X_val) + len(X_test)

    if verbose:
        # Print the shapes of the data
        print("---------------- Training data ----------------")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        print("\n---------------- Validation data ----------------")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        print("\n---------------- Testing data ----------------")
        print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        print("\n---------------- Dataset info ----------------")
        print(f"Number of classes: {num_classes}")
        print(f"Number of features: {num_features}")
        print(f"Number of samples in dataset: {num_samples}")
        print(
            f"Number of samples in training set: {len(X_train)}, "
            f"which is {100 * len(X_train) / num_samples:.2f}% of the dataset"
        )
        print(
            f"Number of samples in validation set: {len(X_val)}, "
            f"which is {100 * len(X_val) / num_samples:.2f}% of the dataset"
        )
        print(
            f"Number of samples in testing set: {len(X_test)}, "
            f"which is {100 * len(X_test) / num_samples:.2f}% of the dataset"
        )

    return num_features, num_classes, num_samples


def exponential_moving_average(data: np.ndarray, alpha: float) -> np.ndarray:
    """Calculate the exponential moving average of a data sequence.

    Args:
        data (np.ndarray): The input data sequence.
        alpha (float): The smoothing factor for the EMA.

    Returns:
        np.ndarray: The exponential moving average of the input data.
    """
    ema = np.zeros(data.size)
    ema[0] = data[0]
    for i in range(1, data.size):
        weights = np.flip((1 - alpha) ** np.arange(i + 1))
        weighted_sum = np.sum(weights * data[: i + 1])
        ema[i] = weighted_sum / np.sum(weights)
    return ema


def dynamic_ema(data: np.ndarray) -> np.ndarray:
    """Calculate the exponential moving average with a dynamically adjusted alpha.

    Args:
        data (np.ndarray): The input data sequence.

    Returns:
        np.ndarray: The exponential moving average of the input data using a dynamic alpha.

    Note:
        The alpha value is dynamically calculated based on the standard deviation of the
        data over a moving window, enhancing the EMA's responsiveness to changes in data variance.
    """
    mean_std = mean_standard_deviation(data, 100)
    alpha = 2 / (mean_std + 1)
    print(f"alpha = {alpha}")
    return exponential_moving_average(data, alpha)


def mean_standard_deviation(data: np.ndarray, window: int) -> float:
    """Compute the mean of the standard deviations over a moving window.

    Args:
        data (np.ndarray): The input data sequence.
        window (int): The size of the moving window.

    Returns:
        float: The maximum standard deviation over the moving window.

    Note:
        This function is used to dynamically adjust the smoothing factor for the EMA calculation.
    """
    stds = np.zeros(data.size)
    for i in range(data.size):
        stds[i] = np.std(data[max(0, i - window) : i + 1])
    return np.max(stds)


def plot_weights_as_templates(weights: np.ndarray, class_labels: list):
    w = linear_classifier.params["W"].data
    w = w.reshape(32, 32, 3, 10)

    # w_min, w_max = torch.min(w), torch.max(w)

    classes = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for c in range(num_classes):
        class_vec = w[:, :, :, c].squeeze()
        w_min, w_max = torch.min(class_vec), torch.max(class_vec)
        wimg = 255.0 * (w[:, :, :, c].squeeze() - w_min) / (w_max - w_min)
        wimg = wimg.type(torch.uint8).numpy()
        axes.flat[c].imshow(wimg)
        axes.flat[c].axis("off")
        axes.flat[c].set_title(classes[c])
