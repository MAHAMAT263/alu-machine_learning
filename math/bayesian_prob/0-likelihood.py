#!/usr/bin/env python3
"""
calculates the likelihood
"""

import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining data given various hypothetical
    probabilities of developing severe side effects.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        text = "x must be an integer that is greater than or equal to 0"
        raise ValueError(text)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    denom = (np.math.factorial(x) * np.math.factorial(n - x))
    binomial_coeff = np.math.factorial(n) / denom
    likelihoods = binomial_coeff * np.power(P, x) * np.power(1 - P, n - x)

    return likelihoods
