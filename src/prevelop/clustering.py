### Prevelop
### Author: Kaspar Bunne

import pandas as pd
import gower
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from collections import Counter


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