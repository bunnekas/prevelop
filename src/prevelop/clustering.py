### Prevelop
### Author: Kaspar Bunne

import pandas as pd
import gower
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import math
import hdbscan
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score


def gower_distance(data):
    """
    Calculate the Gower distance matrix for the given data.

    The Gower distance is a metric that can handle mixed types of data (numerical, categorical, etc.).
    It is particularly useful for clustering tasks where the dataset contains different types of variables.

    Parameters:
    data (pd.DataFrame): A pandas DataFrame containing the data for which the Gower distance matrix is to be calculated.

    Returns:
    np.ndarray: A distance matrix where each element (i, j) represents the Gower distance between the i-th and j-th samples.
    """
    ### calculate distance matrix with gower distance
    distance_gower = gower.gower_matrix(data)
    return distance_gower.astype(np.float64)


def plot_dendrogram(distance_matrix, **kwargs):
    """
    Plots a dendrogram for hierarchical clustering using a given distance matrix.

    Parameters:
    distance_matrix (array-like): The distance matrix used to perform hierarchical clustering.
    **kwargs: Additional keyword arguments passed to the `dendrogram` function from scipy.cluster.hierarchy.

    Returns:
    None

    This function performs hierarchical clustering using the AgglomerativeClustering class from sklearn.
    It computes the linkage matrix and plots the dendrogram using matplotlib and scipy.
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    model = clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0, metric='precomputed', linkage='average', compute_distances=True).fit(distance_matrix)
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    # Plot the corresponding dendrogram
    plt.figure(figsize=(10,10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(linkage_matrix, **kwargs)


def elbow_plot_agglomerative(data_preprocessed, distance_gower, min_clusters, max_clusters):
    """
    Plots an elbow plot for agglomerative clustering using the Gower distance.

    Parameters:
    data_preprocessed (pd.DataFrame): The preprocessed data used for clustering.
    distance_gower (np.ndarray): The precomputed Gower distance matrix.
    min_cluster (int): The minimum number of clusters to consider.
    max_clusters (int): The maximum number of clusters to consider.

    Returns:
    None
    """
    # Validate input
    if min_clusters < 2:
        raise ValueError("min_cluster must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be greater than or equal to min_cluster.")
    if distance_gower.shape[0] != len(data_preprocessed):
        raise ValueError("distance_gower size must match the number of data points in data_preprocessed.")
    
    # Initialize list to store cluster variance
    cluster_variance_average = []

    for k in range(min_clusters, max_clusters + 1):  # Start with min_cluster
        clustering = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='average', compute_distances=True)
        labels = clustering.fit_predict(distance_gower)

        # Compute variance for each cluster
        cluster_variance = 0
        for cluster_label in np.unique(labels):
            cluster_indices = np.where(labels == cluster_label)[0]
            cluster_data = data_preprocessed.iloc[cluster_indices]
            cluster_center = cluster_data.mean(axis=0)  # Cluster centroid
            distances = np.linalg.norm(cluster_data - cluster_center, axis=1)  # Euclidean distances
            cluster_variance += distances.sum()
        
        cluster_variance_average.append(cluster_variance)

    # Plot the elbow plot
    plt.figure(figsize=(16, 8))
    plt.plot(range(min_clusters, max_clusters + 1), cluster_variance_average, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Within-Cluster Variance')
    plt.title(f'Elbow Method for Optimal Clusters (Linkage: Average) [{min_clusters} to {max_clusters}]')
    plt.grid(True)
    if max_clusters - min_clusters <= 50:
        plt.xticks(range(min_clusters, max_clusters + 1, 1))
    elif max_clusters - min_clusters > 50 and max_clusters - min_clusters <= 100:
        plt.xticks(range(min_clusters, max_clusters + 1, 2))
    elif max_clusters - min_clusters > 100 and max_clusters - min_clusters <= 200:
        plt.xticks(range(min_clusters, max_clusters + 1, 5))
    elif max_clusters - min_clusters > 200 and max_clusters - min_clusters <= 500:
        plt.xticks(range(min_clusters, max_clusters + 1, 10))
    elif max_clusters - min_clusters > 500 and max_clusters - min_clusters <= 1000:
        plt.xticks(range(min_clusters, max_clusters + 1, 20))

    # make x-axis labels readable
    plt.xticks(rotation=45)
    plt.title(f'Elbow Method for Optimal Clusters (Linkage: Average) [{min_clusters} to {max_clusters}]')
    plt.show()


def elbow_plot_kmedoids(data_preprocessed, min_clusters, max_clusters):
    """
    Plots the Elbow Method for K-Medoids clustering with customizable range and additional metrics.

    Parameters:
    data_preprocessed (pd.DataFrame): The preprocessed data to be clustered.
    min_cluster (int): The minimum number of clusters to consider.
    max_clusters (int): The maximum number of clusters to consider.

    Returns:
    None
    """
    # Validate input
    if min_clusters < 2:
        raise ValueError("min_cluster must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be greater than or equal to min_cluster.")

    # Initialize lists to store metrics
    wcss = []  # Within-cluster sum of squares (inertia)
    silhouette_scores = []  # Silhouette score
    calinski_harabasz_scores = []  # Calinski-Harabasz index

    for i in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=i, random_state=42)
        kmedoids.fit(data_preprocessed)
        labels = kmedoids.labels_

        # Append within-cluster sum of squares (WCSS)
        wcss.append(kmedoids.inertia_)

        # Calculate Silhouette Score
        if i > 1:  # Silhouette score requires at least 2 clusters
            silhouette_scores.append(silhouette_score(data_preprocessed, labels))
        else:
            silhouette_scores.append(None)

        # Calculate Calinski-Harabasz Index
        calinski_harabasz_scores.append(calinski_harabasz_score(data_preprocessed, labels))

    # Plot the elbow plot with additional metrics
    plt.figure(figsize=(16, 8))

    # Plot WCSS (Inertia)
    plt.plot(range(min_clusters, max_clusters + 1), wcss, marker='o', linestyle='--', label='WCSS (Inertia)')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Inertia)')
    plt.title(f'Elbow Method for Optimal Clusters (K-Medoids) [{min_clusters} to {max_clusters}]')
    if max_clusters - min_clusters <= 50:
        plt.xticks(range(min_clusters, max_clusters + 1, 1))
    elif max_clusters - min_clusters > 50 and max_clusters - min_clusters <= 100:
        plt.xticks(range(min_clusters, max_clusters + 1, 2))
    elif max_clusters - min_clusters > 100 and max_clusters - min_clusters <= 200:
        plt.xticks(range(min_clusters, max_clusters + 1, 5))
    elif max_clusters - min_clusters > 200 and max_clusters - min_clusters <= 500:
        plt.xticks(range(min_clusters, max_clusters + 1, 10))
    elif max_clusters - min_clusters > 500 and max_clusters - min_clusters <= 1000:
        plt.xticks(range(min_clusters, max_clusters + 1, 20))
    plt.grid(True)
    plt.legend()

    plt.show()


def silhouette_score_kmedoids(data_preprocessed, min_clusters, max_clusters):
    """
    Plots Silhouette Scores and Calinski-Harabasz Scores for K-Medoids clustering with a logarithmic y-axis.

    Parameters:
    data_preprocessed (pd.DataFrame): The preprocessed data to be clustered.
    min_cluster (int): The minimum number of clusters to consider.
    max_clusters (int): The maximum number of clusters to consider.

    Returns:
    None
    """
    # Validate input
    if min_clusters < 2:
        raise ValueError("min_cluster must be at least 2.")
    if max_clusters < min_clusters:
        raise ValueError("max_clusters must be greater than or equal to min_cluster.")

    # Initialize lists to store metrics
    silhouette_avg = []  # Silhouette scores

    for i in range(min_clusters, max_clusters + 1):
        kmedoids = KMedoids(n_clusters=i, random_state=42)
        labels = kmedoids.fit_predict(data_preprocessed)

        # Append Silhouette Score
        silhouette_avg.append(silhouette_score(data_preprocessed, labels))

    # Plot the scores
    plt.figure(figsize=(16, 8))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_avg, marker='o', linestyle='--', label='Silhouette Score')
    
    # Set logarithmic y-axis
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title(f'SilhouetteAnalysis for Optimal Clusters (K-Medoids) [{min_clusters} to {max_clusters}]')
    if max_clusters - min_clusters <= 50:
        plt.xticks(range(min_clusters, max_clusters + 1, 1))
    elif max_clusters - min_clusters > 50 and max_clusters - min_clusters <= 100:
        plt.xticks(range(min_clusters, max_clusters + 1, 2))
    elif max_clusters - min_clusters > 100 and max_clusters - min_clusters <= 200:
        plt.xticks(range(min_clusters, max_clusters + 1, 5))
    elif max_clusters - min_clusters > 200 and max_clusters - min_clusters <= 500:
        plt.xticks(range(min_clusters, max_clusters + 1, 10))
    elif max_clusters - min_clusters > 500 and max_clusters - min_clusters <= 1000:
        plt.xticks(range(min_clusters, max_clusters + 1, 20))
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.show()


# def silhouette_plot_kmedoids(data_preprocessed, cluster_sizes):
#     """
#     Generates silhouette plots for different cluster sizes using KMeans clustering.

#     Parameters:
#     data_preprocessed (array-like or DataFrame): The preprocessed data to be clustered.
#     cluster_sizes (list of int): A list of integers representing the different numbers of clusters to evaluate.

#     Returns:
#     None: This function creates plots with silhouette scores for the specified cluster sizes.
#     """
#     # Calculate the number of rows and columns for subplots
#     num_clusters = len(cluster_sizes)
#     cols = 2  # Number of plots per row
#     rows = math.ceil(num_clusters / cols)

#     # Create subplots dynamically
#     fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
#     axes = axes.flatten()  # Flatten axes array for easy iteration

#     for idx, n_clusters in enumerate(cluster_sizes):
#         # Create KMeans instance
#         km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=100, random_state=42)
        
#         # Create SilhouetteVisualizer instance
#         visualizer = SilhouetteVisualizer(km, ax=axes[idx], colors='yellowbrick')
        
#         # Fit the visualizer to data
#         visualizer.fit(data_preprocessed)
        
#         # Set the subplot title
#         axes[idx].set_title(f'Silhouette Plot: {n_clusters} Clusters')

#     # Hide any unused subplots
#     for i in range(len(cluster_sizes), len(axes)):
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()


def agglomerative_clustering(distance_matrix, nr_cluster):
    """
    Perform agglomerative clustering on a given distance matrix.

    Parameters:
    distance_matrix (array-like of shape (n_samples, n_samples)): 
        Precomputed distance matrix.
    nr_cluster (int): 
        The number of clusters to find.

    Returns:
    labels (ndarray of shape (n_samples,)): 
        Cluster labels for each point.
    """
    # function for calulating clusters with agglomerative clustering for given nr of clusters
    clustering = AgglomerativeClustering(n_clusters=nr_cluster,  metric='precomputed', linkage='average', compute_distances=True).fit(distance_matrix)
    return clustering.labels_


def kmedoids_clustering(distance_matrix, nr_cluster):
    """
    Perform k-medoids clustering on a given distance matrix.

    Parameters:
    distance_matrix (array-like of shape (n_samples, n_samples)): 
        Precomputed distance matrix where distance_matrix[i, j] represents 
        the distance between the ith and jth samples.
    nr_cluster (int): 
        The number of clusters to form.

    Returns:
    array of shape (n_samples,): 
        Cluster labels for each point.
    """
    # function for calulating clusters with kmedoids clustering for given nr of clusters
    clustering = KMedoids(n_clusters=nr_cluster, metric='precomputed').fit(distance_matrix)
    return clustering.labels_
    

# def first_clusters(data, distance_matrix, nr_clusters):
#     """
#     Show the first k clusters with the smallest distance using agglomerative clustering.

#     Parameters:
#     data (pd.DataFrame): The input data containing the samples.
#     distance_matrix (np.ndarray): The precomputed distance matrix.
#     nr_clusters (int): The number of clusters to find.

#     Returns:
#     None: This function prints the clusters and their indices in the data.

#     The function performs the following steps:
#     1. Sorts the distance matrix and removes zero distances.
#     2. Iteratively performs agglomerative clustering with increasing distance thresholds.
#     3. Adds clusters with size greater than 1 to the cluster list.
#     4. Prints the clusters and their corresponding indices in the data.
#     """
#     # show the first k clusters with the smallest distance
#     cluster = []
#     sort = np.sort(distance_matrix, axis=None)
#     sort = sort[sort != 0]
#     min_distance = sort[0]
#     t = sort[1]-sort[0]
#     while len(cluster) <= nr_clusters:
#         # show the first clusters to be build by agglomerative clustering
#         clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=min_distance, metric='precomputed', linkage='average', compute_distances=True).fit(distance_matrix)
#         # add cluster with size > 1 to cluster list
#         for i in range(len(Counter(clustering.labels_))):
#             if Counter(clustering.labels_)[i] > 1 and i not in cluster:
#                 cluster.append(i)
#         min_distance += t
#     # print the cluster and their index in the data
#     for i in cluster:
#         print('Cluster:', i)
#         print(data[clustering.labels_ == i].index.tolist())
#         print('------------------------')


def hdbscan_clustering(distance_matrix, min_cluster_size=5, min_samples=None):
    """
    Apply HDBSCAN clustering using a precomputed Gower distance matrix.

    Parameters:
    distance_matrix (np.ndarray): The precomputed Gower distance matrix.
    min_cluster_size (int): The minimum size of clusters. Default is 5.
    min_samples (int): The minimum samples for core points. If None, defaults to min_cluster_size.

    Returns:
    labels (array): Cluster labels assigned by HDBSCAN.
    clusterer (HDBSCAN object): The trained HDBSCAN model.
    """
    clusterer = hdbscan.HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples
    )
    labels = clusterer.fit_predict(distance_matrix)
    return labels, clusterer


def tune_min_cluster_size(distance_matrix, min_size_range, min_samples=None):
    """
    Tune the min_cluster_size parameter for HDBSCAN using a precomputed Gower distance matrix.

    Parameters:
    distance_matrix (np.ndarray): The precomputed Gower distance matrix.
    min_size_range (list of int): Range of min_cluster_size values to test.
    min_samples (int): Minimum samples for core points. If None, defaults to min_cluster_size.

    Returns:
    dict: A dictionary mapping min_cluster_size to the number of clusters detected.
    """
    cluster_counts = {}

    for min_size in min_size_range:
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_size,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(distance_matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
        cluster_counts[min_size] = n_clusters

    # Plot results
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
    Tune the min_samples parameter for HDBSCAN using a precomputed Gower distance matrix.

    Parameters:
    distance_matrix (np.ndarray): The precomputed Gower distance matrix.
    min_samples_range (list of int): Range of min_samples values to test.
    min_cluster_size (int): The minimum size of clusters. Default is 5.

    Returns:
    dict: A dictionary mapping min_samples to the number of clusters detected.
    """
    cluster_counts = {}

    for min_samples in min_samples_range:
        clusterer = hdbscan.HDBSCAN(
            metric="precomputed",
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        labels = clusterer.fit_predict(distance_matrix)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Exclude noise
        cluster_counts[min_samples] = n_clusters

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(min_samples_range, list(cluster_counts.values()), marker="o", linestyle="--")
    plt.xlabel("min_samples")
    plt.ylabel("Number of Clusters")
    plt.title("Tuning min_samples (HDBSCAN)")
    plt.grid()
    plt.show()

    return cluster_counts
