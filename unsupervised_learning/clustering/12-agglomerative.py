#!/usr/bin/env python3
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt
import numpy as np


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset and plots a dendrogram.

    Args:
        X: np.ndarray of shape (n, d) containing the dataset
        dist: maximum cophenetic distance for all clusters

    Returns:
        clss: np.ndarray of shape (n,) containing the cluster indices for each data point
    """
    # Step 1: Compute linkage matrix using Ward linkage
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    # Step 2: Create dendrogram with color threshold
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.axhline(y=dist, color='red', linestyle='--')  # visual cutoff
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Data Point Index')
    plt.ylabel('Cophenetic Distance')
    plt.tight_layout()
    plt.show()

    # Step 3: Form clusters using fcluster based on cophenetic distance
    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    return clss
