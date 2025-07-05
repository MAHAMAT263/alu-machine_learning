#!/usr/bin/env python3
"""
Finds the best number of clusters for a GMM using the Bayesian Information Criterion (BIC).

Uses expectation_maximization from '8-EM' to fit the GMMs.

Usage:
    best_k, best_result, l, b = BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False)

Returns:
    best_k: best number of clusters based on BIC
    best_result: tuple (pi, m, S) of the best model parameters
    l: log likelihoods for all tested k
    b: BIC values for all tested k
    or None, None, None, None on failure
"""

import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Computes BIC to find the best number of clusters for a GMM.

    Args:
        X (np.ndarray): Dataset of shape (n, d)
        kmin (int): Minimum number of clusters (inclusive)
        kmax (int): Maximum number of clusters (inclusive), defaults to number of samples if None
        iterations (int): Max EM iterations
        tol (float): EM tolerance
        verbose (bool): Whether to print EM info

    Returns:
        best_k (int or None), best_result (tuple or None), l (np.ndarray or None), b (np.ndarray or None)
    """
    # Validate inputs
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    n, d = X.shape

    if not isinstance(kmin, int) or kmin < 1 or kmin > n:
        return None, None, None, None

    if kmax is None:
        kmax = n
    if not isinstance(kmax, int) or kmax < kmin or kmax > n:
        return None, None, None, None

    if not isinstance(iterations, int) or iterations < 1:
        return None, None, None, None

    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None

    if not isinstance(verbose, bool):
        return None, None, None, None

    ks = range(kmin, kmax + 1)
    l = []
    b = []
    results = []

    for k in ks:
        pi, m, S, g, log_likelihoods = expectation_maximization(
            X, k, iterations=iterations, tol=tol, verbose=verbose)
        if pi is None:
            return None, None, None, None

        # Number of parameters:
        # pi: k-1 parameters (sum to 1)
        # m: k * d parameters
        # S: k * d * (d+1) / 2 (covariance matrices are symmetric)
        p = (k - 1) + k * d + k * (d * (d + 1) / 2)
        final_log_likelihood = log_likelihoods[-1]
        l.append(final_log_likelihood)
        bic_value = p * np.log(n) - 2 * final_log_likelihood
        b.append(bic_value)
        results.append((pi, m, S))

    l = np.array(l)
    b = np.array(b)

    best_idx = np.argmin(b)
    best_k = kmin + best_idx
    best_result = results[best_idx]

    return best_k, best_result, l, b
