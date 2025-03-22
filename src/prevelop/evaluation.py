### Prevelop
### Author: Kaspar Bunne

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples, silhouette_score
from sklearn.manifold import TSNE


def feature_importance(data, labels):
    """
    Analyze feature importance using a Random Forest classifier, aggregating one-hot encoded features.

    Parameters:
    data (pd.DataFrame): The dataset used for clustering.
    labels (array-like): Cluster labels assigned to each point.

    Returns:
    None: Displays a bar plot of feature importances.
    """
    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(data, labels)
    importances = model.feature_importances_

    # Extract base feature names from one-hot encoded feature names
    base_features = {}
    for feature in data.columns:
        base_feature = feature.split('_')[0]  # Assuming the format is 'feature_a', 'feature_b', etc.
        if base_feature not in base_features:
            base_features[base_feature] = []
        base_features[base_feature].append(feature)

    # Aggregate importance scores for each base feature
    aggregated_importances = {}
    for base_feature, encoded_features in base_features.items():
        aggregated_importances[base_feature] = sum(importances[data.columns.get_loc(f)] for f in encoded_features)

    # Convert aggregated importances to a DataFrame for easier plotting
    aggregated_importances_df = pd.DataFrame(list(aggregated_importances.items()), columns=['Feature', 'Importance'])

    # Sort the DataFrame by importance for better visualization
    aggregated_importances_df = aggregated_importances_df.sort_values(by='Importance', ascending=False)

    # Plot aggregated feature importances
    plt.figure(figsize=(16, 20))
    plt.barh(aggregated_importances_df['Feature'], aggregated_importances_df['Importance'], color="skyblue")
    plt.title("Aggregated Feature Importance for Clustering")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.show()


def compute_dunn_index(data, labels):
    """
    Compute the Dunn Index for the given clustering results.

    Parameters:
    data (array-like): Dataset used for clustering.
    labels (array-like): Cluster labels.

    Returns:
    float: The Dunn Index (higher is better).
    """
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        raise ValueError("Dunn Index requires at least two clusters.")

    # Compute inter-cluster distances
    centroids = [data[labels == label].mean(axis=0) for label in unique_labels if label != -1]
    inter_cluster_distances = np.min([
        np.linalg.norm(c1 - c2)
        for i, c1 in enumerate(centroids)
        for c2 in centroids[i + 1:]
    ])

    # Compute intra-cluster distances
    intra_cluster_distances = np.max([
        np.max(np.linalg.norm(data[labels == label] - centroid, axis=1))
        for label, centroid in zip(unique_labels, centroids)
        if label != -1
    ])

    # Dunn Index
    dunn_index = inter_cluster_distances / intra_cluster_distances
    return dunn_index


def compute_silhouette_score(data, labels):
    """
    Compute the Silhouette Score for the given clustering results.

    Parameters:
    data (array-like): Dataset used for clustering.
    labels (array-like): Cluster labels.

    Returns:
    float: The average Silhouette Score (higher is better).
    """
    if len(set(labels)) < 2:
        raise ValueError("Silhouette Score requires at least two clusters.")

    # Compute the average Silhouette Score
    avg_silhouette = silhouette_score(data, labels)
    return avg_silhouette


def evaluate_clustering(data, labels):
    
    scores = {}
    scores['dbi'] = davies_bouldin_score(data, labels)
    scores['ch-score'] = calinski_harabasz_score(data, labels)
    scores["dunn-index"] = compute_dunn_index(data, labels)
    scores["silhouette-score"] = compute_silhouette_score(data, labels)
    
    return scores


def plot_results_2d(data, labels):
    """
    Plot clustering results in 2D using t-SNE for dimensionality reduction.

    Parameters:
    data (array-like): High-dimensional data.
    labels (array): Cluster labels.
    title (str): Title of the plot.

    Returns:
    None
    """
    # Reduce dimensions for visualization
    tsne = TSNE(n_components=2, random_state=42)
    data_2d = tsne.fit_transform(data)

    plt.figure(figsize=(10,10))
    sns.scatterplot(
        x=data_2d[:, 0],
        y=data_2d[:, 1],
        hue=labels,
        palette="tab10",
        style=(labels == -1).astype(int),  # Different style for noise
        markers=["o", "X"],
        legend="full"
    )
    plt.title("Clustering Results in 2D")
    plt.xlabel("t-SNE Feature 1")
    plt.ylabel("t-SNE Feature 2")
    plt.legend(title="Cluster")
    plt.show()

