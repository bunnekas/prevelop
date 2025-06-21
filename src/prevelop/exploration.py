### PrEvelOp
### Exploratory data analysis module
### Author: Kaspar Bunne

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def boxplots(data, columns):
    """
    Plot boxplots for specified columns.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Column names to plot.

    Returns:
    None
    """
    num_cols = 4
    num_rows = (len(columns) + num_cols - 1) // num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    ax = ax.flatten()

    for i, column in enumerate(columns):
        sns.boxplot(x=data[column], ax=ax[i])
        ax[i].set_title(f'Boxplot: {column}')

    for j in range(len(columns), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def violinplots(data, columns):
    """
    Plot violin plots for specified columns.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Column names to plot.

    Returns:
    None
    """
    num_cols = 4
    num_rows = (len(columns) + num_cols - 1) // num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    ax = ax.flatten()

    for i, column in enumerate(columns):
        sns.violinplot(x=data[column], ax=ax[i])
        ax[i].set_title(f'Violin Plot: {column}')

    for j in range(len(columns), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def distributions(data, columns):
    """
    Plot distribution histograms with KDE for specified columns.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Column names to plot.

    Returns:
    None
    """
    num_cols = 4
    num_rows = (len(columns) + num_cols - 1) // num_cols

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    ax = ax.flatten()

    for i, column in enumerate(columns):
        sns.histplot(data[column], kde=True, ax=ax[i])
        ax[i].set_title(f'Distribution: {column}')

    for j in range(len(columns), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def heatmap(data, columns, method='pearson', figsize=(10, 8), annot=True,
            cmap='coolwarm', title="Correlation Heatmap"):
    """
    Plot a correlation heatmap for the given data.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Columns to include in correlation.
    method (str): Correlation method ('pearson', 'kendall', 'spearman'). Default is 'pearson'.
    figsize (tuple): Figure size. Default is (10, 8).
    annot (bool): Whether to annotate cells. Default is True.
    cmap (str): Colormap. Default is 'coolwarm'.
    title (str): Plot title.

    Returns:
    None
    """
    corr_matrix = data[columns].corr(method=method)

    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt='.2f', cmap=cmap, cbar=True)
    plt.title(title, fontsize=16)
    plt.show()


def z_score_analysis(data, columns, threshold=3, include_outliers=True):
    """
    Identify outliers using Z-Score analysis.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Columns to analyze.
    threshold (float): Z-score threshold for outlier detection. Default is 3.
    include_outliers (bool): If True, return outliers. If False, return non-outliers.

    Returns:
    pd.DataFrame: Filtered rows based on outlier detection.
    """
    z_scores = np.abs((data[columns] - data[columns].mean()) / data[columns].std())
    outliers = (z_scores > threshold).any(axis=1)
    return data[outliers] if include_outliers else data[~outliers]


def isolation_forest(data, columns, contamination=0.05, include_outliers=True, random_state=42):
    """
    Detect outliers using Isolation Forest.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Feature columns for detection.
    contamination (float): Expected proportion of outliers. Default is 0.05.
    include_outliers (bool): If True, return outliers. If False, return non-outliers.
    random_state (int): Random seed. Default is 42.

    Returns:
    pd.DataFrame: Filtered rows based on outlier detection.
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    outlier_labels = iso_forest.fit_predict(data[columns])
    is_outlier = outlier_labels == -1
    return data[is_outlier] if include_outliers else data[~is_outlier]


def tsne_visualization(data, columns, perplexity=30, n_iter=300):
    """
    Visualize data using t-SNE dimensionality reduction.

    Parameters:
    data (pd.DataFrame): Input data.
    columns (list of str): Feature columns for t-SNE.
    perplexity (int): t-SNE perplexity parameter. Default is 30.
    n_iter (int): Number of optimization iterations. Default is 300.

    Returns:
    None
    """
    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(data[columns])
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
