### Prevelop
### Author: Kaspar Bunne

import pandas as pd
import gower
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer


def gower_distance(data):
    ### calculate distance matrix with gower distance
    distance_gower = gower.gower_matrix(data)
    return distance_gower


def plot_dendrogram(distance_matrix, **kwargs):
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


def first_clusters(data, distance_matrix, nr_clusters):
    # show the first k clusters with the smallest distance
    cluster = []
    sort = np.sort(distance_matrix, axis=None)
    sort = sort[sort != 0]
    min_distance = sort[0]
    t = sort[1]-sort[0]
    while len(cluster) <= nr_clusters:
        # show the first clusters to be build by agglomerative clustering
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=min_distance, metric='precomputed', linkage='average', compute_distances=True).fit(distance_matrix)
        # add cluster with size > 1 to cluster list
        for i in range(len(Counter(clustering.labels_))):
            if Counter(clustering.labels_)[i] > 1 and i not in cluster:
                cluster.append(i)
        min_distance += t
    # print the cluster and their index in the data
    for i in cluster:
        print('Cluster:', i)
        print(data[clustering.labels_ == i].index.tolist())
        print('------------------------')


def find_clusters_kmedoids(data_preprocessed, nr_clusters):
    # apply elbow and shilouette method to preprocessed dataframe to find the optimal number of clusters
    wcss = [] 
    silhouette_avg = []
    for i in range(2, nr_clusters): 
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(data_preprocessed) 
        # wcss score
        wcss.append(kmeans.inertia_)
        # silhouette score
        silhouette_avg.append(silhouette_score(data_preprocessed, kmeans.fit_predict(data_preprocessed)))

    # subplot with two plots
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    xticks = range(1, nr_clusters)
    # plot wcss score
    ax[0].plot(range(2, nr_clusters), wcss, marker='o', linestyle='--')
    # plot silhouette score
    ax[1].plot(range(2, nr_clusters), silhouette_avg, marker='o', linestyle='--')
    # make x-labels
    ax[0].set_xlabel('Number of clusters')
    ax[1].set_xlabel('Number of clusters')
    # make y-labels
    ax[0].set_ylabel('WCSS')
    ax[1].set_ylabel('Silhouette score')
    # make title
    ax[0].set_title('Elbow Method')
    ax[1].set_title('Silhouette Method')
    # set xticks
    ax[0].set_xticks(xticks)
    ax[1].set_xticks(xticks)
    # show plot
    plt.show()


def silhouette_plot(data_preprocessed, cluster_sizes):
    fig, ax = plt.subplots(1,4, figsize=(15,8))
    q = 0
    for i in cluster_sizes:
        # Create KMeans instances for different number of clusters
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        # Create SilhouetteVisualizer instance with KMeans instance
        # Fit the visualizer
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q])
        # title for subplot
        ax[q].set_title('Silhouette plot for '+str(i)+' clusters')
        visualizer.fit(data_preprocessed)
        q += 1