#!/usr/bin/env python3
"""Neural network with one hidden layer performing binary classification."""
import numpy as np


class NeuralNetwork:
    """A neural network with one hidden layer performing binary
    classification."""

    def __init__(self, nx, nodes):
        """Initialize the neural network.

        Args:
            nx: The number of input features.
            nodes: The number of nodes found in the hidden layer.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """The weights of the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """The bias of the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """The activated output of the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """The weights of the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """The bias of the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """The activated output of the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """Calculate the forward propagation of the neural network.

        Args:
            X: numpy.ndarray of shape (nx, m) holding the input data.

        Returns:
            The activated outputs of the hidden layer and of the output
            neuron, respectively.
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

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
        """Evaluate the neural network's predictions.

        Args:
            X: numpy.ndarray of shape (nx, m) holding the input data.
            Y: numpy.ndarray of shape (1, m) holding the correct labels.

        Returns:
            The predicted labels, of shape (1, m), and the cost.
        """
        _, A2 = self.forward_prop(X)
        return np.where(A2 >= 0.5, 1, 0), self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculate one pass of gradient descent on the neural network.

        Args:
            X: numpy.ndarray of shape (nx, m) holding the input data.
            Y: numpy.ndarray of shape (1, m) holding the correct labels.
            A1: The output of the hidden layer.
            A2: The predicted output.
            alpha: The learning rate.
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.matmul(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m
        dZ1 = np.matmul(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.matmul(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
