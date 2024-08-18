### Prevelop
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.cluster import AgglomerativeClustering
import gower
from scipy.cluster.hierarchy import dendrogram
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.metrics import silhouette_score


### data preparation, data cleaning, data selection

def load_data(file):
    ### load data and specify key
    # check if file is csv or excel
    if file.endswith('.csv'):
        data = pd.read_csv(file)
        return data
    elif file.endswith('.xlsx'):
        data = pd.read_excel(file)
        return data
    else:
        print('File format not supported')


def clear_cad_data(data):
    ### clear cad-data: drop duplicates, drop rows with missing values, rename columns, delete substrings from columns
    # rename column Benennung (CAD) to Zeichnung
    data = data[['Benennung (CAD)', 'Klasse', 'Volumen [mm3]', 'Masse [kg]',
       'Flächeninhalt [mm2]', 'L [mm]', 'B [mm]', 'H [mm]', 'Lrot [mm]',
       'Da max. [mm]', 'Di min. [mm]']]
    if 'Benennung (CAD)' in data.columns:
        data = data.rename(columns={"Benennung (CAD)": "Zeichnung"})
        data = data.dropna(subset=["Zeichnung"])
    columns = ['Volumen [mm3]','Masse [kg]','Flächeninhalt [mm2]','L [mm]','B [mm]', 'H [mm]','Lrot [mm]','Da max. [mm]','Di min. [mm]']
    # if col in columns is not in data.columns
    for col in columns:
        if col in data.columns:
            # delete substings mm3, kg, mm2, mm from columns Volumen, Masse, Flächeninhalt, L, B, H, Lrot, Da max., Di min.
            data[col] = data[col].str.replace('mm3', '')
            data[col] = data[col].str.replace('kg', '')
            data[col] = data[col].str.replace('mm2', '')
            data[col] = data[col].str.replace('mm', '')
            data[col] = data[col].str.replace(' ', '')
            data[col] = data[col].str.replace(',', '.')
            data[col] = data[col].astype(float)
    # drop rows with missing values in columns L [mm], B [mm] or H [mm]
    data.dropna(subset=['L [mm]', 'B [mm]', 'H [mm]'], inplace=True)
    return data 


def select_data(cad_data, process_data, link_data, key_cad, key_process):
    ### select rows for which cad-data and process-data is available
    # select rows from cad_data with key_cad in column key_cad in link_data
    cad_data = cad_data[cad_data[key_cad].isin(link_data[key_cad])]
    zeichnungen = cad_data[key_cad].tolist()
    # select rows from process_data with process_data in column process_data in link_data
    process_data = process_data[process_data[key_process].isin(link_data[key_process])]
    teile = process_data[key_process].tolist()
    # select rows from link_data with key_cad in zeichnungen list 
    link_data = link_data[link_data[key_cad].isin(zeichnungen)]
    # select rows from data with key_process in teile list
    link_data = link_data[link_data[key_process].isin(teile)]
    # remove duplicates
    link_data = link_data.drop_duplicates()
    zeichnungen = link_data[key_cad].tolist()
    teile = link_data[key_process].tolist()
    cad_data = cad_data[cad_data[key_cad].isin(zeichnungen)]
    process_data = process_data[process_data[key_process].isin(teile)]
    return cad_data, process_data, link_data


def merge_data(cad_data, process_data, key_merge, key_new):
    # natural join data with cad_data on column key
    data = process_data.join(cad_data.set_index(key_merge), on=key_merge)
    # drop colum key_merge
    data.drop(columns=[key_merge], inplace=True)
    # set key_new
    data.set_index(key_new, inplace=True)
    return data



### data exploration




### data preprocessing

def preprocessing(data, num_columns, bin_columns, cat_columns):
    ### preprocess data: scale numerical columns, encode categorical columns
    # split the dataframe in with respect to the selected columns
    df_num = data[num_columns]
    df_bin = data[bin_columns]
    df_cat = data[cat_columns]
    # scale the numerical columns with MaxAbsScaler
    scaler = MaxAbsScaler().fit(df_num)
    df_num_scaled = pd.DataFrame(data=scaler.transform(df_num), index=df_num.index, columns=df_num.columns)
    # encode the categorical columns
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df_cat)
    df_cat_encoded = pd.DataFrame(data=enc.transform(df_cat).toarray(), index=df_cat.index, columns=enc.get_feature_names_out())
    # concatenate the subdataframes columnwise
    data_preprocessed = pd.concat([df_num_scaled, df_cat_encoded, df_bin], axis=1)
    return data_preprocessed



### clustering

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

    
    
    
