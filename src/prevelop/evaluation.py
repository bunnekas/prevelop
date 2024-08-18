### Prevelop
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt
import gower
from sklearn.cluster import AgglomerativeClustering
from yellowbrick.cluster import SilhouetteVisualizer
import seaborn as sns
from collections import Counter


def elbow_plot(data_preprocessed, distance_gower, max_clusters):
    ### plot elbow plot for agglomerative clustering
    # calculate cluster variance for different number of clusters
    cluster_variance_average = []
    for k in range(1,max_clusters):
        clustering = AgglomerativeClustering(n_clusters=k,  metric='precomputed', linkage='average', compute_distances=True).fit(distance_gower)
        cluster_variance_sum = []
        for i in range(k):
            cluster_indices = []
            for index, cluster in enumerate(list(clustering.labels_)):
                if cluster == i:
                    cluster_indices.append(index)
            cluster_len = len(cluster_indices)
            cluster_df = data_preprocessed.iloc[cluster_indices]
            cluster_df.loc['metoid'] = cluster_df.mean()
            cluster_variance = sum(gower.gower_matrix(cluster_df)[-1])
            cluster_variance_sum.append(cluster_variance)
        cluster_variance_average.append(sum(cluster_variance_sum))
    # plot elbow plot for linkage='average'
    plt.figure(figsize=(20, 6), dpi=80)
    plt.plot(range(1,max_clusters), cluster_variance_average[:max_clusters-1])
    plt.xlabel('number of clusters')
    plt.ylabel('sum of within-cluster variance')
    plt.title('linkage: average')
    # make x labels vertical
    plt.xticks(rotation=90)
    plt.xticks(range(1,max_clusters))
    plt.grid()
    plt.show()


# def silhouette_plot(distance_matrix, cluster_sizes):
#     ### plot silhouette plot for agglomerative clustering
#     l = len(cluster_sizes)
#     fig, ax = plt.subplots(1, l, figsize=(15,8))
#     q = 0
#     for i in cluster_sizes:
#         # Create AgglomerativeClustering instances for different number of clusters
#         clustering = AgglomerativeClustering(n_clusters=i,  metric='precomputed', linkage='average', compute_distances=True).fit(distance_matrix)
#         # Fit the visualizer
#         visualizer = SilhouetteVisualizer(clustering, colors='yellowbrick', ax=ax[q])
#         # title for subplot
#         ax[q].set_title('Silhouette plot for '+str(i)+' clusters')
#         visualizer.fit(distance_matrix)
#         q += 1