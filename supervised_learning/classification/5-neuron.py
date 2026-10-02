#!/usr/bin/env python3
"""Single neuron performing binary classification."""
import numpy as np


class Neuron:
    """A single neuron performing binary classification."""

    def __init__(self, nx):
        """Initialize the neuron.

        Args:
            nx: The number of input features to the neuron.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be a integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """The weights vector of the neuron."""
        return self.__W

    @property
    def b(self):
        """The bias of the neuron."""
        return self.__b

    @property
    def A(self):
        """The activated output of the neuron."""
        return self.__A

    def forward_prop(self, X):
        """Calculate the forward propagation of the neuron.

        Args:
            X: numpy.ndarray of shape (nx, m) holding the input data.

        Returns:
            The activated output of the neuron, of shape (1, m).
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """Calculate the cost of the model using logistic regression.

        Args:
            Y: numpy.ndarray of shape (1, m) holding the correct labels.
            A: numpy.ndarray of shape (1, m) holding the activated outputs.

        Returns:
            The cost.
        """
        m = Y.shape[1]
        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m

    def evaluate(self, X, Y):
        """Evaluate the neuron's predictions.

        Args:
            X: numpy.ndarray of shape (nx, m) holding the input data.
            Y: numpy.ndarray of shape (1, m) holding the correct labels.

        Returns:
            The predicted labels, of shape (1, m), and the cost.
        """
        A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculate one pass of gradient descent on the neuron.

        Args:
            X: numpy.ndarray of shape (nx, m) holding the input data.
            Y: numpy.ndarray of shape (1, m) holding the correct labels.
            A: numpy.ndarray of shape (1, m) holding the activated outputs.
            alpha: The learning rate.
        """
        m = Y.shape[1]
        dZ = A - Y
        self.__W = self.__W - alpha * np.matmul(dZ, X.T) / m
        self.__b = self.__b - alpha * np.sum(dZ) / m
