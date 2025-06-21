### PrEvelOp
### Clustering module for distance computation and cluster analysis
### Author: Kaspar Bunne

import pandas as pd
import gower
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import hdbscan


def gower_distance(data):
    """
    Calculate the Gower distance matrix for the given data.

    The Gower distance handles mixed types of data (numerical, categorical)
    and is particularly useful for clustering tasks with heterogeneous features.

    Parameters:
    data (pd.DataFrame): Input data for distance computation.

    Returns:
    np.ndarray: Symmetric distance matrix of shape (n_samples, n_samples).
    """
    distance_gower = gower.gower_matrix(data)
    return distance_gower.astype(np.float64)


def plot_dendrogram(distance_matrix, **kwargs):
    """
    Plot a dendrogram for hierarchical clustering using a precomputed distance matrix.

    Parameters:
    distance_matrix (np.ndarray): Precomputed distance matrix.
    **kwargs: Additional keyword arguments passed to scipy.cluster.hierarchy.dendrogram.

    Returns:
    None
    """
    ### fit agglomerative clustering to get linkage structure
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0,
        metric='precomputed', linkage='average',
        compute_distances=True,
    ).fit(distance_matrix)

    ### compute sample counts under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    ### plot dendrogram
    plt.figure(figsize=(10, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Distance')
    dendrogram(linkage_matrix, **kwargs)


def elbow_plot_agglomerative(data_preprocessed, distance_gower, min_clusters, max_clusters):
    """
    Plot the elbow method for agglomerative clustering using Gower distance.

    Parameters:
    data_preprocessed (pd.DataFrame): Preprocessed data for variance computation.
    distance_gower (np.ndarray): Precomputed Gower distance matrix.
    min_clusters (int): Minimum number of clusters to evaluate.
    max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
    None
    """
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be >= min_clusters.")

    cluster_variance = []
    for k in range(min_clusters, max_clusters + 1):
        clustering = AgglomerativeClustering(
            n_clusters=k, metric='precomputed',
            linkage='average', compute_distances=True,
        )
        labels = clustering.fit_predict(distance_gower)

        ### compute within-cluster variance
        variance = 0
        for cluster_label in np.unique(labels):
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_data = data_preprocessed.iloc[cluster_indices]
            cluster_center = cluster_data.mean(axis=0)
            distances = np.linalg.norm(cluster_data - cluster_center, axis=1)
            variance += distances.sum()
        cluster_variance.append(variance)

    ### plot
    k_range = range(min_clusters, max_clusters + 1)
    plt.figure(figsize=(16, 8))
    plt.plot(k_range, cluster_variance, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Within-Cluster Variance')
    plt.title(f'Elbow Method (Agglomerative, Average Linkage) [{min_clusters}-{max_clusters}]')
    plt.grid(True)
    plt.xticks(list(k_range), rotation=45)
    plt.show()


def elbow_plot_kmedoids(data_preprocessed, min_clusters, max_clusters):
    """
    Plot the elbow method for K-Medoids clustering.

    Parameters:
    data_preprocessed (pd.DataFrame): Preprocessed data for clustering.
    min_clusters (int): Minimum number of clusters to evaluate.
    max_clusters (int): Maximum number of clusters to evaluate.

    Returns:
    None
    """
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be >= min_clusters.")

    wcss = []
    for i in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=i, random_state=42)
        kmedoids.fit(data_preprocessed)
        wcss.append(kmedoids.inertia_)

    ### plot
    k_range = range(min_clusters, max_clusters + 1)
    plt.figure(figsize=(16, 8))
    plt.plot(k_range, wcss, marker='o', linestyle='--', label='WCSS (Inertia)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Inertia)')
    plt.title(f'Elbow Method (K-Medoids) [{min_clusters}-{max_clusters}]')
    plt.xticks(list(k_range), rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()


def silhouette_plot_kmedoids(data_preprocessed, min_clusters, max_clusters):
    """
    Plot silhouette scores for K-Medoids clustering across a range of cluster counts.

    Parameters:
    data_preprocessed (pd.DataFrame): Preprocessed data for clustering.
    min_clusters (int): Minimum number of clusters.
    max_clusters (int): Maximum number of clusters.

    Returns:
    None
    """
    if min_clusters < 2:
        raise ValueError("min_clusters must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be >= min_clusters.")

    silhouette_avg = []
    for i in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=i, random_state=42)
        labels = kmedoids.fit_predict(data_preprocessed)
        silhouette_avg.append(silhouette_score(data_preprocessed, labels))

    ### plot
    k_range = range(min_clusters, max_clusters + 1)
    plt.figure(figsize=(16, 8))
    plt.plot(k_range, silhouette_avg, marker='o', linestyle='--', label='Silhouette Score')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title(f'Silhouette Analysis (K-Medoids) [{min_clusters}-{max_clusters}]')
    plt.xticks(list(k_range), rotation=45)
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


def agglomerative_clustering(distance_matrix, nr_cluster):
    """
    Perform agglomerative clustering on a precomputed distance matrix.

    Parameters:
    distance_matrix (np.ndarray): Precomputed distance matrix.
    nr_cluster (int): Number of clusters to form.

    Returns:
    np.ndarray: Cluster labels for each sample.
    """
    clustering = AgglomerativeClustering(
        n_clusters=nr_cluster, metric='precomputed',
        linkage='average', compute_distances=True,
    ).fit(distance_matrix)
    return clustering.labels_


def kmedoids_clustering(distance_matrix, nr_cluster):
    """
    Perform K-Medoids clustering on a precomputed distance matrix.

    Parameters:
    distance_matrix (np.ndarray): Precomputed distance matrix.
    nr_cluster (int): Number of clusters to form.

    Returns:
    np.ndarray: Cluster labels for each sample.
    """
    clustering = KMedoids(
        n_clusters=nr_cluster, metric='precomputed',
    ).fit(distance_matrix)
    return clustering.labels_


def hdbscan_clustering(distance_matrix, min_cluster_size=5, min_samples=None):
    """
    Apply HDBSCAN clustering using a precomputed distance matrix.

    Parameters:
    distance_matrix (np.ndarray): Precomputed distance matrix.
    min_cluster_size (int): Minimum size of clusters. Default is 5.
    min_samples (int): Minimum samples for core points. Defaults to min_cluster_size.

    Returns:
    tuple:
        - labels (np.ndarray): Cluster labels (-1 indicates noise).
        - clusterer (hdbscan.HDBSCAN): Trained HDBSCAN model.
    """
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    labels = clusterer.fit_predict(distance_matrix)
    return labels, clusterer


def tune_min_cluster_size(distance_matrix, min_size_range, min_samples=None):
    """
    Tune the min_cluster_size parameter for HDBSCAN.

    Parameters:
    distance_matrix (np.ndarray): Precomputed distance matrix.
    min_size_range (list of int): Range of min_cluster_size values to test.
    min_samples (int): Minimum samples for core points.

    Returns:
    dict: Mapping of min_cluster_size to number of clusters detected.
    """
    cluster_counts = {}
    for min_size in min_size_range:
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_size,
            min_samples=min_samples,
        )
        labels = clusterer.fit_predict(distance_matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_counts[min_size] = n_clusters

    ### plot results
    plt.figure(figsize=(8, 5))
    plt.plot(min_size_range, list(cluster_counts.values()), marker="o", linestyle="--")
    plt.xlabel("min_cluster_size")
    plt.ylabel("Number of Clusters")
    plt.title("Tuning min_cluster_size (HDBSCAN)")
    plt.grid()
    plt.show()

    return cluster_counts


def tune_min_samples(distance_matrix, min_samples_range, min_cluster_size=5):
    """
    Tune the min_samples parameter for HDBSCAN.

    Parameters:
    distance_matrix (np.ndarray): Precomputed distance matrix.
    min_samples_range (list of int): Range of min_samples values to test.
    min_cluster_size (int): Minimum size of clusters. Default is 5.

    Returns:
    dict: Mapping of min_samples to number of clusters detected.
    """
    cluster_counts = {}
    for min_samples in min_samples_range:
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
        )
        labels = clusterer.fit_predict(distance_matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        cluster_counts[min_samples] = n_clusters

    ### plot results
    plt.figure(figsize=(8, 5))
    plt.plot(min_samples_range, list(cluster_counts.values()), marker="o", linestyle="--")
    plt.xlabel("min_samples")
    plt.ylabel("Number of Clusters")
    plt.title("Tuning min_samples (HDBSCAN)")
    plt.grid()
    plt.show()

    return cluster_counts
