#!/usr/bin/env python3
""" Module to implement Neuron class
"""
import numpy as np


class Neuron:
    """ Class that implements a neuron that performs
    binary classification
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def __sigmoid(self, Z):
        """ Calculates the sigmoid of a number or numpy array
        Args:
            Z (numpy.ndarray): input either as a number or numpy.ndarray
        Return:
            Z (numpy.ndarray | numpy.float)
        """
        return 1 / (1 + np.exp(-Z))

    def forward_prop(self, X):
        """ Calculates the forward propagation for X using the
        sigmoid activation function
        Args:
            X (numpy.ndarray): input in the shape (nx, m) where nx is
            the number of input features and m is the number of
            training examples
        Returns:
            A: The activation
        """
        # z = wx + b
        Z = self.__W @ X + self.__b

        # a = sigmoid(z)
        A = self.__sigmoid(Z)

        self.__A = A

        return A

    def cost(self, Y, A):
        """ Calculates the cost of the model.
        Args:
            Y (numpy.ndarray) with shape (1, m) the correct output labels
            A (numpy.ndarray) with shape (1, m) the predicted output labels
        Returns:
            (numpy.float) the cost of the model
        """
        _, m = Y.shape

        costs = Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)
        cost = (-1 / m) * np.sum(costs)
        return cost
