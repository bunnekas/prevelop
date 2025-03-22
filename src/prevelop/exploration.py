### Prevelop
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pandas.plotting import parallel_coordinates, andrews_curves


def boxplots(data, columns):
    """
    Plot boxplots for specified columns in the given DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data to be plotted.
    columns (list of str): A list of column names to plot boxplots for.

    Returns:
    None
    """
    num_cols = 4  # Number of boxplots per row
    num_rows = (len(columns) + num_cols - 1) // num_cols  # Calculate rows dynamically

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    ax = ax.flatten()  # Flatten axes array for easy indexing

    for i, column in enumerate(columns):
        sns.boxplot(x=data[column], ax=ax[i])
        ax[i].set_title(f'Boxplot for {column}')
    
    # Hide unused subplots
    for j in range(len(columns), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def violinplots(data, columns):
    """
    Plot violin plots for specified columns in the given DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data to be plotted.
    columns (list of str): A list of column names to plot violin plots for.

    Returns:
    None
    """
    num_cols = 4  # Number of plots per row
    num_rows = (len(columns) + num_cols - 1) // num_cols  # Calculate rows dynamically

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    ax = ax.flatten()  # Flatten the axes array for easier indexing

    for i, column in enumerate(columns):
        sns.violinplot(x=data[column], ax=ax[i])
        ax[i].set_title(f'Violin Plot for {column}')
    
    # Hide unused subplots
    for j in range(len(columns), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


def distributions(data, columns):
    """
    Plot the distribution for specified columns in the given DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): The list of column names for which the distribution plots are to be created.

    Returns:
    None
    """
    num_cols = 4  # Number of plots per row
    num_rows = (len(columns) + num_cols - 1) // num_cols  # Calculate rows dynamically

    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
    ax = ax.flatten()  # Flatten the array of axes for easy indexing

    for i, column in enumerate(columns):
        sns.histplot(data[column], kde=True, ax=ax[i])
        ax[i].set_title(f'Distribution for {column}')
    
    # Hide unused subplots
    for j in range(len(columns), len(ax)):
        ax[j].set_visible(False)

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()


# def barplots(data, columns):
#     """
#     Plot barplots for specified categorical columns in the given DataFrame.

#     Parameters:
#     data (pd.DataFrame): The DataFrame containing the data.
#     columns (list of str): A list of column names to plot barplots for.

#     Returns:
#     None
#     """
#     num_cols = 2  # Number of barplots per row
#     num_rows = (len(columns) + num_cols - 1) // num_cols  # Calculate rows dynamically

#     fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, num_rows * 4))
#     ax = ax.flatten()  # Flatten axes to ensure consistent 1D indexing

#     for i, column in enumerate(columns):
#         sns.barplot(
#             x=data[column].value_counts().index,
#             y=data[column].value_counts().values,
#             ax=ax[i]
#         )
#         ax[i].set_title(f'Barplot for {column}')

#     # Hide unused subplots
#     for j in range(len(columns), len(ax)):
#         ax[j].axis('off')

#     plt.tight_layout()
#     # make xtixk labels vertical
#     plt.xticks(rotation=45)
#     plt.show()


def heatmap(data, columns, method='pearson', figsize=(10, 8), annot=True, cmap='coolwarm', title="Correlation Heatmap"):
    """
    Plot a heatmap for the correlation matrix of the given DataFrame.

    Parameters:
    data (pd.DataFrame): The DataFrame for which to compute the correlation heatmap.
    method (str): The method for correlation ('pearson', 'kendall', 'spearman'). Default is 'pearson'.
    figsize (tuple): The size of the heatmap figure. Default is (12, 10).
    annot (bool): Whether to annotate the heatmap cells. Default is True.
    cmap (str): Colormap for the heatmap. Default is 'coolwarm'.
    title (str): The title of the heatmap. Default is "Correlation Heatmap".

    Returns:
    None
    """
    # Compute correlation matrix
    corr_matrix = data[columns].corr(method=method)

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, fmt='.2f', cmap=cmap, cbar=True)
    plt.title(title, fontsize=16)
    plt.show()


def z_score_analysis(data, columns, threshold=3, include_outliers=True):
    """
    Identify outliers using Z-Score Analysis.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): Columns to perform Z-score analysis on.
    threshold (float): The Z-score threshold to identify outliers. Default is 3.
    include_outliers (bool): Whether to return only outliers (True) or non-outliers (False). Default is True.

    Returns:
    pd.DataFrame: A DataFrame containing rows identified as outliers or non-outliers.
    """
    # Calculate Z-scores
    z_scores = np.abs((data[columns] - data[columns].mean()) / data[columns].std())

    # Identify rows with any Z-score above the threshold
    outliers = (z_scores > threshold).any(axis=1)

    # Return outliers or non-outliers based on the include_outliers flag
    return data[outliers] if include_outliers else data[~outliers]


def isolation_forest(data, columns, contamination=0.05, include_outliers=True, random_state=42):
    """
    Detect outliers using Isolation Forest.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): Columns to use for outlier detection.
    contamination (float): The proportion of outliers in the data. Default is 0.05 (5%).
    include_outliers (bool): Whether to return only outliers (True) or non-outliers (False). Default is True.
    random_state (int): Seed for random number generator. Default is 42.

    Returns:
    pd.DataFrame: DataFrame containing rows identified as outliers or non-outliers.
    """
    # Initialize Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    
    # Fit the model and predict outliers
    outlier_labels = iso_forest.fit_predict(data[columns])
    
    # Outliers are labeled as -1, inliers as 1
    is_outlier = outlier_labels == -1

    # Return outliers or non-outliers based on the flag
    return data[is_outlier] if include_outliers else data[~is_outlier]


def tsne_visualization(data, columns, perplexity=30, n_iter=300):
    """
    Visualize data using t-SNE.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the data.
    columns (list of str): Columns to use for t-SNE visualization.
    perplexity (int): Perplexity parameter for t-SNE.
    n_iter (int): Number of iterations for t-SNE optimization.

    Returns:
    None
    """
    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_results = tsne.fit_transform(data[columns])
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title("t-SNE Visualization")
    plt.show()