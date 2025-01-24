#!/usr/bin/env python3
""" Module to implement Neuron class
"""
import matplotlib.pyplot as plt
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

        self.__W = np.random.normal((1, nx))
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

    def evaluate(self, X, Y):
        """ Evaluates the model
        Args:
            X (numpy.ndarray) with the shape (nx, m) the input data
            Y (numpy.ndarray) with the shape (1, m) the correct labels for
            the data
        Returns:
            (Y_hat, cost) the predictions of the model and the cost
        """
        # Make predictions
        A = self.forward_prop(X)

        # Calculate the cost
        cost = self.cost(Y, A)

        # Categorise predictions
        Y_hat = np.where(A >= 0.5, 1, 0)

        return Y_hat, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron
        Args:
            X (numpy.ndarray)
            Y (numpy.ndarray)
            A (numpy.ndarray)
            alpha (float)
        Returns:
            None
        """
        m = X.shape[1]
        dz = (A - Y)

        dL_dW = (1 / m) * dz @ X.T
        dL_db = (1 / m) * np.sum(dz)

        self.__W -= alpha * dL_dW
        self.__b -= alpha * dL_db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the neuron
        Args:
            X (numpy.ndarray)
            Y (numpy.ndarray)
            iterations (int)
            alpha (float)
        Returns:
            float
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')

        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')

        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')

        if alpha <= 0:
            raise ValueError('alpha must be positive')

        if graph or verbose:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')

            if step <= 0 and step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []

        for i in range(0, iterations):
            self.__A = self.forward_prop(X)

            if iterations % step == 0 and (graph or verbose):
                cost = self.cost(Y, self.__A)

                if verbose:
                    print('Cost after {} iterations: {}'.format(i, cost))

                if graph:
                    costs.append(cost)
            
            self.gradient_descent(X, Y, self.__A, alpha)

        # Calculate the final cost after training
        self.__A = self.forward_prop(X)
        cost = self.cost(Y, self.__A)
        costs.append(cost)

        # Print final cost after training if verbose
        if verbose:
            print('Cost after {} iterations: {}'.format(iterations, cost))

        # Create plot
        if graph:
            plt.title('Training Cost')
            plt.xlabel = 'iterations'
            plt.ylabel = 'cost'
            plt.plot(np.arange(0, iterations, 1), cost)
            plt.show()

        return self.evaluate(X, Y)
