### Prevelop
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import matplotlib.pyplot as plt


def boxplots(data, columns):
    ### plot boxplots for columns in data
    # create figure with subplots, plot 5 columns in one row
    fig, ax = plt.subplots(1, len(columns), figsize=(20, 6))
    # plot boxplots for columns in data
    for i, column in enumerate(columns):
        sns.boxplot(y=data[column], ax=ax[i])
        ax[i].set_title('Boxplot for '+column, rotation=45)
        # hide axis labels
        ax[i].set_ylabel(None)
    plt.show()  


def distribution(data, columns):
    ### plot distribution for columns in data
    # create figure with subplots
    fig, ax = plt.subplots(1, len(columns), figsize=(20, 6))
    # plot distribution for columns in data
    for i, column in enumerate(columns):
        sns.histplot(data[column], ax=ax[i])
        ax[i].set_title('Distribution for '+column)
    plt.show()
