from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class LinearClassifier:
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        reg: float = 1e-3,
        num_iters: int = 1000,
    ):

        self.num_classes = num_classes
        self.num_features = num_features

        self.reg = reg
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.params = dict(
            W=nn.Parameter(
                torch.randn(num_features, num_classes, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
        )

    def train(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
    ) -> tuple:

        # Initialize the best validation accuracy and the best parameters
        best_val_acc = 0
        best_params = dict()

        # Initialize the loss and accuracy history
        loss_history = dict(train=dict(), val=dict())
        acc_history = dict(train=dict(), val=dict())

        # Training loop
        for i in tqdm(range(self.num_iters), desc="Training"):

            # Select a random batch of data
            batch_indices = torch.randint(0, X_train.shape[0], (self.batch_size,))
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]

            # Zero the gradients
            self._zero_gradients()

            # Compute the loss and backpropagate
            train_loss = self.loss(X_batch, y_batch)
            train_loss.backward(retain_graph=True)
            self._update_weights()

            # Save the training loss
            loss_history["train"][i] = train_loss.data

            # Every 500 iterations, compute the validation loss and accuracy
            if i % 100 == 0 or i == self.num_iters - 1:

                # Compute the validation loss
                with torch.no_grad():
                    val_loss = self.loss(X_val, y_val)
                loss_history["val"][i] = val_loss.data

                # Predict the labels for the training and validation data
                y_pred_train = self.predict(X_train)
                y_pred_val = self.predict(X_val)

                # Compute the training and validation accuracy from the predicted labels
                acc_history["train"][i] = accuracy_score(y_train, y_pred_train)
                acc_history["val"][i] = accuracy_score(y_val, y_pred_val)

                # If the current validation accuracy is the best so far, save the parameters
                if acc_history["val"][i] > best_val_acc:
                    best_val_acc = acc_history["val"][i]
                    # best_params = deepcopy(self.params)
                    best_params['W'] = self.params['W'].clone()
                    best_params['b'] = self.params['b'].clone()

        # Update the parameters with the best ones
        self.params = best_params

        return loss_history, acc_history

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute the logits of the model.

        Args:
            X: Input data of shape (N, D)

        Returns:
            logits: The logits of the model. Tensor of shape (N, C)

        """
        logits = torch.zeros((X.shape[0], self.num_classes))

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement computation of the logits of the model.                 #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        logits = X @ self.params['W'] + self.params['b']

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return logits

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the labels of the data.

        Args:
            X (torch.Tensor): Input data of shape (N, D)

        Returns:
            y_pred (torch.Tensor): The predicted labels of the data. Array of shape (N,)
        """

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement the prediction function of the model.                   #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        logits = self.forward(X)
        y_pred = torch.argmax(logits, axis=1)

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return y_pred

    def loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the loss of the model.

        Args:
            X (torch.Tensor): Input data of shape (N, D)
            y (torch.Tensor): Labels of shape (N,)

        Returns:
            torch.Tensor: The loss of the model
        """

        loss = torch.tensor([0.0], requires_grad=True)

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.3 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement the loss function of the model. The loss function       #
        # should be the cross-entropy loss with L2 regularization.          #
        #                                                                   #
        # HINT 1: You may find torch.nn.CrossEntropyLoss() useful.          #
        # HINT 2: Use the self.reg attribute for the regularization term.   #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        
        y.to(torch.float)
        y_pred = self.forward(X)
        y_pred.to(torch.float)
        l = torch.nn.CrossEntropyLoss()
        loss = l(y_pred, y)

        l2_reg = torch.sum(self.params['b'] ** 2)
        l2_reg += torch.sum(self.params['W'] ** 2)

        loss = loss + self.reg * l2_reg

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return loss

    def _update_weights(self):
        """Update the weights of the model using the gradient descent."""

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 3.4 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement the loss function of the model. The loss function       #
        # should be the cross-entropy loss with L2 regularization.          #
        #                                                                   #
        # HINT 1: You may find torch.nn.CrossEntropyLoss() useful.          #
        # HINT 2: Use the self.learning_rate attribute for the              #
        # learning rate.                                                    #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        
        if self.params['W'].grad is not None:
            self.params['W'].data = self.params['W'] - self.learning_rate * self.params['W'].grad

        if self.params['b'].grad is not None:
            self.params['b'].data = self.params['b'] - self.learning_rate * self.params['b'].grad

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    def _zero_gradients(self):
        """Zero the gradients of the model parameters."""
        for name in self.params.keys():
            if self.params[name].grad is not None:
                self.params[name].grad.zero_()
