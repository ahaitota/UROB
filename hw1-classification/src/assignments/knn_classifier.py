import numpy as np


class KNNClassifier:
    """K-Nearest Neighbors Classifier

    This class implements a K-Nearest Neighbors (KNN) classifier for classification tasks.

    Parameters:
        k (int, optional): The number of nearest neighbors to consider for classification. Defaults to 1.
        vectorized (bool, optional): If True, use vectorized calculations for distance computation
        for faster execution. Defaults to False.

    Attributes:
        k (int): The number of nearest neighbors to consider for classification.
        vectorized (bool): Indicates whether vectorized calculations for distance computation are enabled.
        X_train (numpy.ndarray or None): The training data features.
        y_train (numpy.ndarray or None): The training data labels.
    """

    def __init__(self, k: int = 1, vectorized: bool = False):
        assert k > 0, "k must be greater than 0!"

        self.k = k
        self.vectorized = vectorized

        self.X_train = None
        self.y_train = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the KNN classifier on the provided training data.

        Args:
            X (numpy.ndarray): The training data features.
            y (numpy.ndarray): The training data labels.

        Returns:
            None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts labels for a given set of data points.

        Args:
            X (numpy.ndarray): The input data points for which labels are to be predicted.

        Returns:
            numpy.ndarray: The predicted labels for the input data points.

        Raises:
            AssertionError: If the classifier has not been trained or if the number
            of features in the training data and input data do not match.
        """

        assert (
            self.X_train is not None and self.y_train is not None
        ), "Train the classifier first!"
        assert (
            self.X_train.shape[1] == X.shape[1]
        ), "Train and test data must have the same number of features!"

        if self.vectorized:
            dists = self._compute_distances_vectorized(X)
        else:
            dists = self._compute_distances(X)

        return self._predict_labels(dists)

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """Computes the L2 distance between test points and training points (non-vectorized).

        Args:
            X (numpy.ndarray): The input data points for which distances are to be computed.

        Returns:
            numpy.ndarray: The computed distances between input data points and training data points.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train), dtype=np.float32)

        for i in range(num_test):
            for j in range(num_train):
                # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 1.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
                # TODO:                                                             #
                # Calculate the L2 distance between the ith test point and the jth  #
                # training point and store the result in dists[i, j]. Avoid using   #
                # loops over dimensions or np.linalg.norm().                        #
                # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
                # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
                
                dists[i][j] = np.sqrt(np.sum((self.X_train[j] - X[i])**2))

                # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return dists

    def _compute_distances_vectorized(self, X: np.ndarray) -> np.ndarray:
        """Computes the L2 distance between test points and training points (vectorized).

        Args:
            X (numpy.ndarray): The input data points for which distances are to be computed.

        Returns:
            numpy.ndarray: The computed distances between input data points and training data points.
        """

        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 1.3 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Calculate the L2 distance between every test point and every      #
        # training point efficiently, without resorting to explicit loops.  #
        # Store the computed distances in a variable called 'dists'.        #
        #                                                                   #
        # To achieve this, implement the function using fundamental array   #
        # operations only. Avoid using functions from the 'scipy' library   #
        # and refrain from utilizing 'np.linalg.norm()'.                    #
        #                                                                   #
        # Hint: Consider formulating the L2 distance calculation through    #
        # matrix multiplication and two broadcasted summations. (You might  #
        # want to look this up on the internet.)                            #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        
        dists = np.sqrt(np.sum(X**2, axis=1).reshape(X.shape[0], 1) + np.sum(self.X_train**2, axis=1) - 2 * X.dot(self.X_train.T))

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return dists

    def _predict_labels(self, dists: np.ndarray) -> np.ndarray:
        """Predicts labels based on the distance matrix.

        Args:
            dists (numpy.ndarray): The distance matrix containing distances between test points and training points.

        Returns:
            numpy.ndarray: The predicted labels for the test data points.
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            y_pred[i] = 0
            # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 1.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
            # TODO:                                                             #
            # Utilize the distance matrix to identify the k nearest neighbors   #
            # of the ith testing point. Then, retrieve the labels of these      #
            # neighbors from the 'self.y_train'. After identifying the labels   #
            # of the k nearest neighbors, determine the most frequent label     #
            # within the list of labels. Save this label as 'y_pred[i]'.        #
            # In case of a tie, select the smaller label as the final choice.   #
            #                                                                   #
            # Hint: You may find the 'numpy.argsort' and 'numpy.bincount'       #
            # functions useful.                                                 #
            # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
            # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)

            y_pred[i] = np.argmax(np.bincount(self.y_train[np.argsort(dists[i])[0 : self.k]]))
            

            # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return y_pred
