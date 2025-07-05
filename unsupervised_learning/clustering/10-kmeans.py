#!/usr/bin/env python3
"""
Performs K-means clustering on a dataset.

Usage:
    C, clss = kmeans(X, k)

Returns:
    C: numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    clss: numpy.ndarray of shape (n,) containing the cluster index for each data point
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means clustering on dataset X into k clusters using sklearn.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d)
        k (int): Number of clusters

    Returns:
        tuple: (C, clss)
            C (numpy.ndarray): Centroid coordinates of shape (k, d)
            clss (numpy.ndarray): Cluster labels for each point of shape (n,)
    """
    kmeans_model = sklearn.cluster.KMeans(n_clusters=k, n_init='auto')
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    return C, clss
