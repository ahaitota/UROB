from typing import Union, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier


class Data2DVisualizer:
    """
    A class for visualizing 2D datasets and their splits, including decision boundaries.

    This class allows you to visualize datasets and their splits, such as training, validation, and test sets.
    It also provides the capability to display decision boundaries for classifiers.

    Args:
        train_split (Tuple[np.ndarray, np.ndarray]):
            A tuple containing the training data features and labels.

        val_split (Tuple[np.ndarray, np.ndarray], optional):
            A tuple containing the validation data features and labels. Defaults to (np.array([]), np.array([])).

        test_split (Tuple[np.ndarray, np.ndarray], optional):
            A tuple containing the test data features and labels. Defaults to (np.array([]), np.array([])).

    Properties:
        x_lim (Tuple[float, float]): Returns the x-axis limits for the plot.
        y_lim (Tuple[float, float]): Returns the y-axis limits for the plot.
        num_classes (int): Returns the number of unique classes in the dataset.
        num_splits (int): Returns the number of dataset splits available.

    Methods:
        show_dataset: Displays a visualization of dataset splits using subplots for training, validation, and test sets.
        show_decision_boundaries: Displays decision boundaries for a classifier.
        show_knn_principle: Illustrates the k-Nearest Neighbors (k-NN) principle with a plot.
    """

    def __init__(
        self,
        train_split: Tuple[
            Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]
        ],
        val_split: Tuple[
            Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]
        ] = (np.array([]), np.array([])),
        test_split: Tuple[
            Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]
        ] = (np.array([]), np.array([])),
    ):
        self.X_train, self.y_train = train_split
        self.X_val, self.y_val = val_split
        self.X_test, self.y_test = test_split

        self.val = True if self.X_test.shape[0] != 0 else False
        self.test = True if self.X_train.shape[0] != 0 else False

        self.features = np.concatenate([self.X_train, self.X_val, self.X_test])
        self.labels = np.concatenate([self.y_train, self.y_val, self.y_test])

        self.colors = [
            [0, 0.4, 1],
            [1, 0, 0.4],
            [0, 1, 0.5],
            [1, 0.7, 0.5],
            "violet",
            "mediumaquamarine",
        ]
        self.color_map = ListedColormap(self.colors[: self.num_classes])

    @property
    def x_lim(self) -> Tuple[float, float]:
        """Get the x-axis limits for the plot.

        Returns:
            Tuple[float, float]: A tuple containing the minimum and maximum values of the x-axis.
        """

        min_x, max_x = np.min(self.features[:, 0]), np.max(self.features[:, 0])
        range_x = max_x - min_x
        return min_x - range_x * 0.1, max_x + range_x * 0.1

    @property
    def y_lim(self) -> Tuple[float, float]:
        """Get the y-axis limits for the plot.

        Returns:
            Tuple[float, float]: A tuple containing the minimum and maximum values of the y-axis.
        """

        min_y, max_y = np.min(self.features[:, 1]), np.max(self.features[:, 1])
        range_y = max_y - min_y
        return min_y - range_y * 0.1, max_y + range_y * 0.1

    @property
    def num_classes(self) -> int:
        """Get the number of unique classes in the dataset.

        Returns:
            int: The number of unique classes.
        """

        return np.size(np.unique(self.labels))

    @property
    def num_splits(self) -> int:
        """Get the number of dataset splits available.

        Returns:
            int: The number of dataset splits (1 + validation + test).
        """

        return 1 + self.val + self.test

    @property
    def lut(self) -> np.ndarray:
        """
        Example for 3 classes
        cmap = np.zeros((255, 4))

        # first third is first color
        cmap[:85, :] = np.array(self.color_map(0)) * 255

        # second third is second color
        cmap[85:170, :] = np.array(self.color_map(1)) * 255

        # last third is third color
        cmap[170:, :] = np.array(self.color_map(2)) * 255
        """

        lut = np.zeros((255, 4))

        step = 255 // self.num_classes

        for i in range(self.num_classes):
            lut[i * step : (i + 1) * step, :] = np.array(self.color_map(i)) * 255

        return lut

    def show_dataset(self) -> None:
        """Displays a visualization of dataset splits using subplots for training,
        validation, and test sets.

        Returns:
            None
        """

        fig, axes = plt.subplots(1, self.num_splits, figsize=(self.num_splits * 6, 6))
        fig.suptitle("Dataset Splits", fontsize=30)

        self._plot_data(self.X_train, self.y_train, axes[0], title="Training")
        self._plot_data(self.X_val, self.y_val, axes[1], title="Validation")
        self._plot_data(self.X_test, self.y_test, axes[2], title="Test")

        plt.tight_layout()
        plt.show()

    def show_decision_boundaries(self, classifier, h: float = 0.001) -> None:
        """Display decision boundaries for a classifier.

        This function plots decision boundaries for a given classifier along with the dataset splits.

        Args:
            classifier: The trained classifier for which decision boundaries will be plotted.
            h (float, optional): Step size for meshgrid. Smaller values create finer boundaries. Default is 0.001.

        Returns:
            None
        """

        fig, axes = plt.subplots(1, self.num_splits, figsize=(self.num_splits * 6, 6))
        fig.suptitle("Decision Boundaries", fontsize=30)

        self._plot_data(self.X_train, self.y_train, axes[0], title="Training")
        self._plot_decision_boundaries(classifier, h, axes[0])

        self._plot_data(self.X_val, self.y_val, axes[1], title="Validation")
        self._plot_decision_boundaries(classifier, h, axes[1])

        self._plot_data(self.X_test, self.y_test, axes[2], title="Test")
        self._plot_decision_boundaries(classifier, h, axes[2])

        plt.tight_layout()
        plt.show()

    def show_knn_principle(self) -> None:
        """Illustrate the k-Nearest Neighbors (k-NN) principle with a plot.

        This function creates a plot demonstrating the k-NN principle using a sample data point and its neighbors.

        Returns:
            None
        """

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_data(self.X_train, self.y_train, ax)

        p = np.array([0.5, 0.5])

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(self.X_train, self.y_train)

        neighbor_indices = knn.kneighbors([p], return_distance=False)[0]

        for i in neighbor_indices:
            ax.plot(
                [p[0], self.X_train[i, 0]],
                [p[1], self.X_train[i, 1]],
                color="gray",
                linestyle="--",
                linewidth=2,
            )
        ax.scatter(
            p[0],
            p[1],
            color="#ffd500",
            s=100,
            linewidth=1.5,
            zorder=10,
            edgecolor="black",
        )

        fig.savefig("knn_principle.png", dpi=70, bbox_inches="tight", pad_inches=0.1)

    def show_linear_weights(self, classifier) -> None:
        """Illustrate the linear classifier weights with a plot.

        This function creates a plot demonstrating the linear classifier weights.

        Returns:
            None
        """

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_decision_boundaries(classifier, 0.01, ax)

        # Set the axis limits based on the maximum of x and y limits (needed for quiver plot)
        min_lim = max(self.x_lim[0], self.y_lim[0])
        max_lim = min(self.x_lim[1], self.y_lim[1])

        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)

        weights = classifier.W.data.T
        biases = classifier.b.data

        xx = np.linspace(min_lim, max_lim, 100)

        for c in range(classifier.num_classes):
            w = weights[c]
            b = biases[c]

            # Calculate the line equation using original weights
            yy = -w[0] / w[1] * xx - b / w[1]
            ax.plot(xx, yy, color=self.colors[c], linewidth=2, zorder=100)
            ax.plot(xx, yy, color="black", linewidth=4, zorder=99)

            # Calculate the direction vector for quiver plot
            v = np.linalg.pinv(w[np.newaxis, :]) * (-b)
            w_normalized = w / np.linalg.norm(w)
            ax.quiver(
                v[0],
                v[1],
                w_normalized[0],
                w_normalized[1],
                color=self.colors[c],
                zorder=1000,
                scale=2,
                scale_units="xy",
                edgecolor="black",
                linewidth=1.5,
            )

            # Add dot at the end of the vector
            ax.scatter(
                v[0],
                v[1],
                color=self.colors[c],
                s=25,
                linewidth=1.5,
                zorder=1000,
                edgecolor="black",
            )

        ax.axis("off")

    def show_decision_functions(self, classifier) -> None:
        """Illustrate the decision functions with a plot.
        This function creates a plot demonstrating the decision functions of a classifier.

        Args:
        classifier: The trained classifier for which decision functions will be plotted.

        Returns:
        None
        """
        x_rng = np.linspace(self.x_lim[0], self.x_lim[1], 100)
        y_rng = np.linspace(self.y_lim[0], self.y_lim[1], 100)
        xx, yy = np.meshgrid(x_rng, y_rng)
        X = np.column_stack((xx.ravel(), yy.ravel()))
        X = torch.from_numpy(X).float()

        with torch.no_grad():
            logits = classifier.forward(X)
        logits = logits.numpy()

        fig = go.Figure()

        # Plot decision surfaces
        for c in range(self.num_classes):
            zz = logits[:, c].reshape(xx.shape)
            color = f"rgb{tuple(int(val * 255) for val in self.colors[c])}"

            fig.add_trace(
                go.Surface(
                    x=xx,
                    y=yy,
                    z=zz,
                    colorscale=[[0, color], [1, color]],
                    opacity=1,
                    showscale=False,
                    name=f"Class {c} Decision Function",
                    showlegend=True,
                )
            )

        # Plot training points
        y_train = (
            self.y_train
            if isinstance(self.y_train, np.ndarray)
            else self.y_train.numpy()
        )
        for c in range(self.num_classes):
            mask = y_train == c
            color = f"rgb{tuple(int(val * 255) for val in self.colors[c])}"

            fig.add_trace(
                go.Scatter3d(
                    x=self.X_train[mask, 0],
                    y=self.X_train[mask, 1],
                    z=np.zeros(np.sum(mask)),
                    mode="markers",
                    marker=dict(
                        size=5,
                        color=color,
                        line=dict(width=1, color="black"),  # width of the border
                    ),
                    name=f"Class {c} Samples",
                )
            )

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Decision Function",
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        )

        fig.show()

    def _plot_data(
        self,
        features: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        ax: plt.Axes,
        title: str = "",
    ):
        """Plot data points with labels.

        This internal function is used to plot data points with labels and adjust the plot settings.

        Parameters:
            features (numpy.ndarray): The features of the data points.
            labels (numpy.ndarray): The labels of the data points.
            ax (matplotlib.pyplot.Axes): The axis for plotting.
            title (str, optional): The title of the plot. Default is an empty string.

        Returns:
            None
        """

        ax.set_xlim(self.x_lim[0], self.x_lim[1])
        ax.set_ylim(self.y_lim[0], self.y_lim[1])
        ax.axis("off")

        scatter_args = dict(
            s=100,
            alpha=0.85,
            c=labels,
            linewidths=1.5,
            edgecolors="black",
            cmap=self.color_map,
            zorder=10,
        )

        ax.scatter(features[:, 0], features[:, 1], **scatter_args)
        ax.set_title(title, fontsize=20)

    def _plot_decision_boundaries(self, classifier, h: float, ax: plt.Axes) -> None:
        """Plot decision boundaries.

        This internal function is used to plot decision boundaries for a given classifier.

        Parameters:
            classifier: The trained classifier for which decision boundaries will be plotted.
            h (float): Step size for meshgrid. Smaller values create finer boundaries.
            ax (matplotlib.pyplot.Axes): The axis for plotting.

        Returns:
            None
        """

        x = np.arange(self.x_lim[0], self.x_lim[1], h)
        y = np.arange(self.y_lim[0], self.y_lim[1], h)
        xx, yy = np.meshgrid(x, y)

        mesh_matrix = np.c_[xx.ravel(), yy.ravel()]

        if isinstance(self.X_train, torch.Tensor):
            mesh_matrix = torch.from_numpy(mesh_matrix).float()
            mesh_predictions = classifier.predict(mesh_matrix)
            mesh_predictions = mesh_predictions.detach().numpy()
        else:
            mesh_predictions = classifier.predict(mesh_matrix)

        mesh_predictions = mesh_predictions.reshape(xx.shape)
        ax.pcolormesh(xx, yy, mesh_predictions, alpha=0.4, cmap=self.color_map)
