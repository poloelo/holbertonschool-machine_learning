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
