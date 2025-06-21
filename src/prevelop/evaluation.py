### PrEvelOp
### Clustering evaluation module
### Author: Kaspar Bunne

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.manifold import TSNE


def feature_importance(data, labels):
    """
    Analyze feature importance using a Random Forest classifier.

    Trains a Random Forest to predict cluster labels, then aggregates
    importance scores for one-hot encoded features back to their base features.

    Parameters:
    data (pd.DataFrame): Dataset used for clustering.
    labels (array-like): Cluster labels assigned to each sample.

    Returns:
    None: Displays a horizontal bar plot of feature importances.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(data, labels)
    importances = model.feature_importances_

    ### aggregate one-hot encoded features back to base features
    base_features = {}
    for feature in data.columns:
        base_feature = feature.split('_')[0]
        if base_feature not in base_features:
            base_features[base_feature] = []
        base_features[base_feature].append(feature)

    aggregated = {}
    for base_feature, encoded_features in base_features.items():
        aggregated[base_feature] = sum(
            importances[data.columns.get_loc(f)] for f in encoded_features
        )

    ### sort and plot
    importance_df = pd.DataFrame(
        list(aggregated.items()), columns=['Feature', 'Importance']
    ).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, max(6, len(importance_df) * 0.8)))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color="skyblue")
    plt.title("Feature Importance for Clustering")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.show()


def compute_dunn_index(data, labels):
    """
    Compute the Dunn Index for clustering results.

    The Dunn Index measures the ratio of the minimum inter-cluster distance
    to the maximum intra-cluster distance. Higher values indicate better clustering.

    Parameters:
    data (array-like): Dataset used for clustering.
    labels (array-like): Cluster labels.

    Returns:
    float: The Dunn Index.
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Dunn Index requires at least two clusters.")

    ### compute cluster centroids (excluding noise label -1)
    valid_labels = [l for l in unique_labels if l != -1]
    centroids = [data[labels == label].mean(axis=0) for label in valid_labels]

    ### minimum inter-cluster distance
    inter_cluster_distances = [
        np.linalg.norm(c1 - c2)
        for i, c1 in enumerate(centroids)
        for c2 in centroids[i + 1:]
    ]
    min_inter = np.min(inter_cluster_distances)

    ### maximum intra-cluster distance
    intra_cluster_distances = [
        np.max(np.linalg.norm(data[labels == label] - centroid, axis=1))
        for label, centroid in zip(valid_labels, centroids)
    ]
    max_intra = np.max(intra_cluster_distances)

    return min_inter / max_intra


def compute_silhouette_score(data, labels):
    """
    Compute the average Silhouette Score for clustering results.

    Parameters:
    data (array-like): Dataset used for clustering.
    labels (array-like): Cluster labels.

    Returns:
    float: Average Silhouette Score (-1 to 1, higher is better).
    """
    if len(set(labels)) < 2:
        raise ValueError("Silhouette Score requires at least two clusters.")
    return silhouette_score(data, labels)


def evaluate_clustering(data, labels):
    """
    Compute a set of clustering evaluation metrics.

    Includes Davies-Bouldin Index, Calinski-Harabasz Score,
    Dunn Index, and Silhouette Score.

    Parameters:
    data (array-like): Dataset used for clustering.
    labels (array-like): Cluster labels.

    Returns:
    dict: Dictionary with metric names as keys and scores as values.
    """
    scores = {}
    scores['dbi'] = davies_bouldin_score(data, labels)
    scores['ch-score'] = calinski_harabasz_score(data, labels)
    scores['dunn-index'] = compute_dunn_index(data, labels)
    scores['silhouette-score'] = compute_silhouette_score(data, labels)
    return scores


def plot_results_2d(data, labels):
    """
    Plot clustering results in 2D using t-SNE dimensionality reduction.

    Parameters:
    data (array-like): High-dimensional data.
    labels (array-like): Cluster labels.

    Returns:
    None
    """
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        hue=labels,
        palette="tab10",
        style=(labels == -1).astype(int),
        markers=["o", "X"],
        legend="full",
    )
    plt.title("Clustering Results (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(title="Cluster")
    plt.show()
