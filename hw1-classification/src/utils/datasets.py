import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(69)


def create_spiral_dataset(
    file: str, num_samples: int, num_classes: int, std: float = 0.2
) -> None:
    """Create a synthetic dataset of spirals and save it to a file.

    This function generates a synthetic dataset consisting of multiple spiral-shaped classes
    and saves it to a file in a NumPy-compatible format.

    Args:
        file (str): The filename (including path) where the dataset will be saved.
        num_samples (int): The total number of data points to generate.
        num_classes (int): The number of classes (spiral arms) in the dataset.
        std (float, optional): The standard deviation of noise added to the data points. Default is 0.2.

    Returns:
        None
    """

    # Initialize arrays for features (X) and labels (y)
    X = np.zeros((num_samples * num_classes, 2))
    y = np.zeros(num_samples * num_classes, dtype=np.int64)
    for c in range(num_classes):
        r_min, r_max = 0.2, 1
        t_min = c * 2 * np.pi / num_classes
        t_max = (c + 2) * 2 * np.pi / num_classes
        r = np.linspace(r_min, r_max, num_samples)
        t = np.linspace(t_min, t_max, num_samples) + np.random.normal(
            0, std, num_samples
        )

        # Define indices for dataset and spiral generation
        spiral_indices = np.arange(num_samples)
        dataset_indices = np.arange(num_samples * c, num_samples * (c + 1))

        # Generate data points for each spiral
        for si, di in zip(spiral_indices, dataset_indices):
            X[di] = r[si] * np.sin(t[si]), r[si] * np.cos(t[si])
            y[di] = c

    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Save the dataset to the specified file
    np.savez(
        file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # Plot the generated data
    fig, ax = plt.subplots()
    for c in range(num_classes):
        ax.scatter(X[y == c, 0], X[y == c, 1], label=f"Spiral {c + 1}")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Scatter Plot of Spirals")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


def create_circle_dataset(
    file: str, num_samples: int, num_classes: int, noise: float = 0.2
) -> None:
    """Create a synthetic dataset of circles and save it to a file.

    This function generates a synthetic dataset consisting of multiple concentric circles
    and saves it to a file in a NumPy-compatible format.

    Args:
        file (str): The filename (including path) where the dataset will be saved.
        num_samples (int): The total number of data points to generate.
        num_classes (int): The number of classes (concentric circles) in the dataset.
        noise (float, optional): The standard deviation of noise added to the data points. Default is 0.2.

    Returns:
        None
    """

    # Initialize arrays for features (X) and labels (y)
    X, y = [], []

    for i in range(num_classes):
        # Generate polar coordinates for each circle
        theta = np.linspace(0, 2 * np.pi, num_samples) + np.random.normal(
            0, noise, num_samples
        )
        r = i + np.random.normal(0, noise, num_samples)

        # Convert polar coordinates to Cartesian coordinates (x, y)
        x = r * np.cos(theta)
        y_vals = r * np.sin(theta)

        # Append generated data points and labels
        X.extend(np.vstack((x, y_vals)).T)
        y.extend([i] * num_samples)

    # Convert lists to NumPy arrays
    X, y = np.array(X), np.array(y)

    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Save the dataset to the specified file
    np.savez(
        file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # Plot the data
    fig, ax = plt.subplots()
    for c in range(num_classes):
        ax.scatter(X[y == c, 0], X[y == c, 1], label=f"Circle {c + 1}")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Scatter Plot of Circles")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


def create_moon_dataset(
    file: str, num_samples: int, noise: float = 0.2, random_state: int = 42
) -> None:
    """Create a synthetic dataset of moons and save it to a file.

    This function generates a synthetic dataset consisting of two interleaving half circles
    and saves it to a file in a NumPy-compatible format.

    Args:
        file (str): The filename (including path) where the dataset will be saved.
        num_samples (int): The total number of data points to generate.
        noise (float, optional): The standard deviation of noise added to the data points. Default is 0.2.
        random_state (int, optional): The random seed to use for reproducibility. Default is 42.

    Returns:
        None
    """

    # Generate the data
    X, y = make_moons(n_samples=num_samples, noise=noise, random_state=random_state)

    # Split the data into training, validation, and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Save the data to a file
    np.savez(
        file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Scatter Plot of Moons")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


def create_linearly_separable_dataset(file: str, num_samples: int) -> None:
    """Create a synthetic linearly separable dataset and save it to a file.

    This function generates a synthetic dataset with three linearly separable classes
    and saves it to a file in a NumPy-compatible format.

    Args:
        file (str): The filename (including path) where the dataset will be saved.
        num_samples (int): The total number of data points to generate.

    Returns:
        None
    """

    # Generate synthetic features with Gaussian noise
    mean_1 = [0, 0]  # Mean for class 1
    cov_1 = [[0.01, 0], [0, 0.1]]  # Covariance matrix for class 1
    class_1_features = np.random.multivariate_normal(mean_1, cov_1, num_samples)

    mean_2 = [1, 0]  # Mean for class 2
    cov_2 = [[0.05, 0], [0, 0.06]]  # Covariance matrix for class 2
    class_2_features = np.random.multivariate_normal(mean_2, cov_2, num_samples)

    mean_3 = [0, 1]  # Mean for class 3
    cov_3 = [[0.11, 0], [0, 0.03]]  # Covariance matrix for class 3
    class_3_features = np.random.multivariate_normal(mean_3, cov_3, num_samples)

    # Create labels for the three classes
    class_1_labels = np.zeros(num_samples, dtype=int)
    class_2_labels = np.ones(num_samples, dtype=int)
    class_3_labels = 2 * np.ones(num_samples, dtype=int)

    # Combine features and labels for all three classes
    X = np.vstack((class_1_features, class_2_features, class_3_features))
    y = np.concatenate((class_1_labels, class_2_labels, class_3_labels))

    # Split the dataset into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

    # Save the data to a file
    np.savez(
        file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )

    # Plot the data
    fig, ax = plt.subplots()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")
    ax.scatter(X[y == 2, 0], X[y == 2, 1], label="Class 2")
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("Scatter Plot of Linearly Separable Classes")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()


if __name__ == "__main__":
    create_spiral_dataset("../data/toy_datasets_2/spiral_dataset", 1000, 3, std=0.2)
    create_circle_dataset("../data/toy_datasets_2/circle_dataset", 1000, 3, noise=0.1)
    create_moon_dataset("../data/toy_datasets_2/moon_dataset", 1000, noise=0.1)
    create_linearly_separable_dataset(
        "../data/toy_datasets_2/linearly_separable_dataset", 500
    )
