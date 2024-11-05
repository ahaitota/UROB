from typing import List, Union
from dataclasses import dataclass

import torch
import numpy as np
import matplotlib.pyplot as plt

from .general import exponential_moving_average


def plot_cifar10(X: np.ndarray, y: np.ndarray) -> None:
    """Display a grid of random samples from the CIFAR-10 dataset.

    Args:
        X (np.ndarray): Array of images with shape (num_samples, height, width, channels).
        y (np.ndarray): Array of labels corresponding to the images.

    Returns:
        None

    Note:
        The function selects 7 random samples from each of the 10 classes in the CIFAR-10 dataset.
        Images are displayed in a grid with the class name shown above the first image of each row.
    """
    generator = torch.Generator().manual_seed(42)
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

    # Reshape the images to 32x32 pixels
    X = X.reshape(-1, 32, 32, 3)

    num_classes = len(classes)
    samples_per_class = 7

    plt.figure(figsize=(num_classes, samples_per_class))

    for label, class_name in enumerate(classes):
        # Find indices of images for the current class
        indices = np.flatnonzero(y == label)
        indices = np.random.choice(indices, samples_per_class, replace=False)

        for i, idx in enumerate(indices):
            plt_idx = i * num_classes + label + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X[idx].astype("uint8"))
            plt.axis("off")
            if i == 0:
                plt.title(class_name)
    plt.show()


def plot_knn_cross_validation(k_to_metrics: dict, label_names: list = None):
    """Show the results of cross-validation for different k values in KNN.

    Args:
        k_to_metrics (dict): A dictionary containing cross-validation results.
            - The dictionary should have keys 'accuracy', 'precision', 'recall', and 'f1'.
            - Each key should map to a dictionary where the keys are k values (as strings) and the values are lists of metric values.
        label_names (List[str], optional): A list of class labels for the precision, recall, and F1 score metrics. Default is None.

    Returns:
        None
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle("Cross-validation on k")

    # Plot accuracy on the first subplot
    k_acc = np.array([int(k) for k in k_to_metrics["accuracy"].keys()])
    acc_means = np.array([np.mean(v) for v in k_to_metrics["accuracy"].values()])
    acc_stds = np.array([np.std(v) for v in k_to_metrics["accuracy"].values()])
    _plot_error_bar(
        ax1,
        k_acc,
        acc_means,
        acc_stds,
        x_label="k",
        y_label="Accuracy",
        title="Accuracy",
    )

    # Plot precision on the second subplot
    k_pre = np.array([int(k) for k in k_to_metrics["precision"].keys()])
    precision_means = np.hstack(
        [v[..., np.newaxis] for v in k_to_metrics["precision"].values()]
    )
    _plot_knn_metric(ax2, k_pre, precision_means, label_names, "Precision")

    # Plot recall on the third subplot
    k_rec = np.array([int(k) for k in k_to_metrics["recall"].keys()])
    recall_means = np.hstack(
        [v[..., np.newaxis] for v in k_to_metrics["recall"].values()]
    )
    _plot_knn_metric(ax3, k_rec, recall_means, label_names, "Recall")

    # Plot f1 on the fourth subplot
    k_f1 = np.array([int(k) for k in k_to_metrics["f1"].keys()])
    f1_means = np.hstack([v[..., np.newaxis] for v in k_to_metrics["f1"].values()])
    _plot_knn_metric(ax4, k_f1, f1_means, label_names, "F1")

    plt.tight_layout()
    plt.show()


def plot_training(
    loss_history: dict, accuracy_history: dict, ema: bool = False, alpha: float = 0.1
) -> None:
    """Show the training history of a neural network.

    Args:
        loss_history (dict): Dictionary containing the training and validation loss history.
        accuracy_history (dict): Dictionary containing the training and validation accuracy history.
        ema (bool, optional): Whether to apply exponential moving average to the history. Defaults to False.
        alpha (float, optional): The alpha value for the exponential moving average. Defaults to 0.1.

    Returns:
        None

    Note:
        This function creates a plot with two subplots, displaying the loss and accuracy history.
    """
    train_loss_iters = np.array(list(loss_history["train"].keys()))
    train_losses = np.array(list(loss_history["train"].values()))

    val_loss_iters = np.array(list(loss_history["val"].keys()))
    val_losses = np.array(list(loss_history["val"].values()))

    train_acc_iters = np.array(list(accuracy_history["train"].keys()))
    train_accs = np.array(list(accuracy_history["train"].values()))

    val_acc_iters = np.array(list(accuracy_history["val"].keys()))
    val_accs = np.array(list(accuracy_history["val"].values()))

    if ema:
        train_losses = exponential_moving_average(train_losses, alpha)
        val_losses = exponential_moving_average(val_losses, alpha)
        train_accs = exponential_moving_average(train_accs, alpha)
        val_accs = exponential_moving_average(val_accs, alpha)

    X_acc = [train_acc_iters, val_acc_iters]
    Y_acc = [train_accs, val_accs]

    X_loss = [train_loss_iters, val_loss_iters]
    Y_loss = [train_losses, val_losses]

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    _plot_values(
        axes[0],
        X_loss,
        Y_loss,
        ["Training", "Validation"],
        "Iterations",
        "Loss",
        "Loss History",
    )
    _plot_values(
        axes[1],
        X_acc,
        Y_acc,
        ["Training", "Validation"],
        "Iterations",
        "Accuracy",
        "Accuracy History",
    )

    plt.show()


def plot_weights_as_templates(weights: torch.Tensor, class_labels: list):
    w = weights.data
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
    for c in range(len(classes)):
        class_vec = w[:, :, :, c].squeeze()
        w_min, w_max = torch.min(class_vec), torch.max(class_vec)
        wimg = 255.0 * (w[:, :, :, c].squeeze() - w_min) / (w_max - w_min)
        wimg = wimg.type(torch.uint8).numpy()
        axes.flat[c].imshow(wimg)
        axes.flat[c].axis("off")
        axes.flat[c].set_title(classes[c])

    plt.show()


def _plot_knn_metric(
    ax: plt.Axes,
    k_choices: np.ndarray,
    metric_means: np.ndarray,
    label_names: List[str],
    metric_name: str,
) -> None:
    """Plot a metric across different values of k.

    Args:
        ax (plt.Axes): The subplot where the metric will be plotted.
        k_choices (np.ndarray): Array of k values.
        metric_means (np.ndarray): 2D array of metric values, with each row corresponding to a class.
        label_names (List[str]): List of labels for each class.
        metric_name (str): The name of the metric to be plotted.

    Returns:
        None
    """
    for i in range(metric_means.shape[0]):
        label = label_names[i] if label_names is not None else f"Class {i}"
        ax.plot(k_choices, metric_means[i], linewidth=1.5, marker="o", label=label)
    _set_plot(ax, k_choices, "k", metric_name, metric_name, legend=True)


def _plot_values(
    ax: plt.Axes,
    X: Union[np.ndarray, List[np.ndarray]],
    Y: Union[np.ndarray, List[np.ndarray]],
    label_names: List[str] = None,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    """Plot data on a given Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        X (Union[np.ndarray, List[np.ndarray]]): The x-coordinates of the data.
        Y (Union[np.ndarray, List[np.ndarray]]): The y-coordinates of the data.
        label_names (List[str], optional): A list of labels for the legend. Default is None.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.
        title (str, optional): The title of the plot. Default is an empty string.

    Returns:
        None

    Note:
        If the x-coordinates have fewer than 20 elements, markers are shown on the plot.
    """
    assert len(X) == len(Y), "X and Y must have the same length"
    legend = label_names is not None and len(X) > 1

    for i in range(len(X)):
        label = label_names[i] if legend else None
        if X[i].size < 20:
            ax.plot(X[i], Y[i], linewidth=1.5, marker="o", label=label)
        else:
            ax.plot(X[i], Y[i], linewidth=1.5, label=label)
    _set_plot(ax, X, x_label, y_label, title, legend=legend)


def _plot_error_bar(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    stds: np.ndarray,
    x_label: str = "",
    y_label: str = "",
    title: str = "",
) -> None:
    """Plot data with error bars on a given Matplotlib Axes object.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to plot on.
        x (np.ndarray): The x-coordinates of the data.
        y (np.ndarray): The y-coordinates of the data.
        stds (np.ndarray): The standard deviations for the error bars.
        x_label (str, optional): The label for the x-axis. Default is an empty string.
        y_label (str, optional): The label for the y-axis. Default is an empty string.
        title (str, optional): The title of the plot. Default is an empty string.

    Returns:
        None
    """
    ax.errorbar(
        x, y, yerr=stds, linestyle="dotted", linewidth=1.5, marker="o", capsize=4
    )
    _set_plot(ax, x, x_label, y_label, title)


def _set_plot(
    ax: plt.Axes,
    X: Union[np.ndarray, List[np.ndarray]],
    x_label: str,
    y_label: str,
    title: str,
    legend: bool = False,
) -> None:
    """Set plot settings such as labels, title, grid, and legend.

    Args:
        ax (plt.Axes): The Matplotlib Axes object to adjust.
        X (Union[np.ndarray, List[np.ndarray]]): The x-coordinates used to determine ticks and limits.
        x_label (str): The label for the x-axis.
        y_label (str): The label for the y-axis.
        title (str): The title of the plot.
        legend (bool): Whether to display a legend.

    Returns:
        None
    """
    unique_x = list(set(np.concatenate(X))) if isinstance(X, list) else X
    min_k, max_k = np.min(unique_x), np.max(unique_x)

    if legend:
        ax.legend()

    if len(unique_x) < 20:
        ax.set_xticks(unique_x)
        ax.set_xlim(min_k - 1, max_k + 0.35 * (max_k - min_k))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True)
