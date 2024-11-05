from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class MLPClassifier:
    def __init__(
        self,
        num_features: int,
        hidden_dim_1: int,
        hidden_dim_2: int,
        num_classes: int,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        reg: float = 1e-6,
        batch_size: int = 100,
        num_iters: int = 1000,
    ):

        self.params = dict(
            W1=nn.Parameter(
                torch.randn(num_features, hidden_dim_1, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b1=nn.Parameter(torch.zeros(hidden_dim_1, dtype=torch.float)),
            W2=nn.Parameter(
                torch.randn(hidden_dim_1, hidden_dim_2, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b2=nn.Parameter(torch.zeros(hidden_dim_2, dtype=torch.float)),
            W3=nn.Parameter(
                torch.randn(hidden_dim_2, num_classes, dtype=torch.float)
                * np.sqrt(2 / (num_features + num_classes))
            ),
            b3=nn.Parameter(torch.zeros(num_classes, dtype=torch.float)),
        )

        self.reg = reg
        self.num_iters = num_iters
        self.activation = activation
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        self.activation_func = None

        if activation == "relu":
            self.activation_func = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_func = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_func = nn.Tanh()

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
            if i % 500 == 0 or i == self.num_iters - 1:

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
                    best_params = deepcopy(self.params)

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

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.1 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement computation of the logits of the model.                 #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        first = self.activation_func(X @ self.params['W1'] + self.params['b1'])
        second = self.activation_func(first @ self.params['W2'] + self.params['b2'])
        logits = second @ self.params['W3'] + self.params['b3']

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

        return logits

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict the labels of the data.

        Args:
            X (torch.Tensor): Input data of shape (N, D)

        Returns:
            y_pred (torch.Tensor): The predicted labels of the data. Array of shape (N,)
        """

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.2 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
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

        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱ Assignment 4.3 ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # TODO:                                                             #
        # Implement the loss function of the model. The loss function       #
        # should be the cross-entropy loss with L2 regularization.          #
        #                                                                   #
        # HINT: You may find torch.nn.CrossEntropyLoss() useful.            #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        y.to(torch.float)
        y_pred = self.forward(X)
        y_pred.to(torch.float)
        l = torch.nn.CrossEntropyLoss()
        loss = l(y_pred, y)

        l2_reg = torch.sum(self.params['b1'] ** 2) + torch.sum(self.params['b2'] ** 2) + torch.sum(self.params['b3'] ** 2)
        l2_reg += torch.sum(self.params['W1'] ** 2) + torch.sum(self.params['W3'] ** 2) + torch.sum(self.params['W3'] ** 2)

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
        # HINT: You may find torch.nn.CrossEntropyLoss() useful.            #
        #                                                                   #
        # Good luck!                                                        #
        # ▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰▱▰ #
        # 🌀 INCEPTION 🌀 (Your code begins its journey here. 🚀 Do not delete this line.)
        
        with torch.no_grad():
            for _, param in self.params.items():
                if param.grad is not None:
                    param -= self.learning_rate * param.grad

        # 🌀 TERMINATION 🌀 (Your code reaches its end. 🏁 Do not delete this line.)

    def _zero_gradients(self):
        for name in self.params.keys():
            if self.params[name].grad is not None:
                self.params[name].grad.zero_()
