### Prevelop
### preprocessing module for data loading and preprocessing
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler


def load_data(file):
    """
    Load data from a CSV or Excel file.

    Parameters:
    file (str): The path to the file to be loaded. The file should be in CSV or Excel (.xlsx) format.

    Returns:
    DataFrame: A pandas DataFrame containing the loaded data if the file format is supported.

    Raises:
    ValueError: If the file format is not supported.
    """
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
    """
    Cleans and preprocesses CAD data by performing the following operations:
    
    1. Drops duplicate rows.
    2. Drops rows with missing values in specific columns.
    3. Renames columns:
       - Renames 'Benennung (CAD)' to 'Zeichnung'.
       - Renames 'Key' to 'Zeichnung' if 'Benennung (CAD)' is not present.
    4. Deletes specific substrings ('mm3', 'kg', 'mm2', 'mm', ' ') from certain columns and converts them to float.
    5. Drops rows with missing values in 'L [mm]', 'B [mm]', or 'H [mm]'.
    
    Parameters:
    data (pandas.DataFrame): The input CAD data to be cleaned.
    
    Returns:
    pandas.DataFrame: The cleaned and preprocessed CAD data.
    """
    columns = data.columns
    # set key
    if 'Benennung (CAD)' in columns:
        data = data.rename(columns={"Benennung (CAD)": "Zeichnung"})
        data = data.dropna(subset=["Zeichnung"])
    elif 'Key' in data.columns:
        data = data.rename(columns={"Key": "Zeichnung"})
        data = data.dropna(subset=["Zeichnung"])
    else:
        raise ValueError("No key found.")
    # remove units
    columns_clean = ['Volumen [mm3]','Masse [kg]','Flächeninhalt [mm2]','L [mm]','B [mm]', 'H [mm]','Lrot [mm]','Da max. [mm]','Di min. [mm]']
    for col in columns_clean:
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
    # fill missing values with 0
    data.fillna(0, inplace=True)
    return data 


def aggregate_data(data, key, columns, methods):
    """
    Aggregates process data based on specified methods for each column.

    Parameters:
    data (pd.DataFrame): The input data to be aggregated.
    key (str): The key column to group by.
    columns (list of str): The list of columns to aggregate.
    methods (list of str): The list of methods to apply for each column. Supported methods are 'encode', 'sum', 'mean', 'max', and 'min'.

    Returns:
    pd.DataFrame: A new DataFrame with aggregated data.

    Raises:
    ValueError: If an unsupported method is provided in the methods list.
    """
    # funtion to aggregate process data
    teile_list = list(set(data[key].tolist()))
    data_new = pd.DataFrame(data=teile_list, columns=[key])
    for column, method in zip(columns, methods):
        column_list = []
        if method == 'encode':
            for teil in teile_list:
                teil_data = data[data[key] == teil]
                column_list.append(teil_data[column].tolist())
            data_new[column] = column_list
            # select all values in column unique
            values = list(set([item for sublist in column_list for item in sublist]))
            for value in values:
                column_values = [1 if value in process else 0 for process in data_new[column]]
                data_new[str(column) + ' ' + str(value)] = column_values
            data_new.drop(columns=[column], inplace=True)
        elif method == 'sum':
            data_new[column] = data.groupby(key)[column].sum().tolist()
        elif method == 'mean':
            data_new[column] = data.groupby(key)[column].mean().tolist()
        elif method == 'max':
            data_new[column] = data.groupby(key)[column].max().tolist()
        elif method == 'min':
            data_new[column] = data.groupby(key)[column].min().tolist()
        else:
            raise ValueError('Method not supported')
    return data_new


def select_data(cad_data, process_data, link_data, key_cad, key_process):
    """
    Selects and filters rows from cad_data, process_data, and link_data based on the availability of 
    corresponding keys in link_data.

    Args:
        cad_data (pd.DataFrame): DataFrame containing CAD data.
        process_data (pd.DataFrame): DataFrame containing process data.
        link_data (pd.DataFrame): DataFrame containing linking information between CAD and process data.
        key_cad (str): Column name in cad_data and link_data used as a key for CAD data.
        key_process (str): Column name in process_data and link_data used as a key for process data.

    Returns:
        tuple: A tuple containing three DataFrames:
            - cad_data (pd.DataFrame): Filtered CAD data.
            - process_data (pd.DataFrame): Filtered process data.
            - link_data (pd.DataFrame): Filtered linking data with duplicates removed.
    """
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


def merge_data(cad_data, process_data, link_data, key_merge, key_new):
    """
    Merges two dataframes on a specified key and sets a new index.

    This function performs a natural join between `cad_data` and `process_data` 
    on the column specified by `key_merge`. After the join, the column `key_merge` 
    is dropped, and the index of the resulting dataframe is set to `key_new`.

    Parameters:
    cad_data (pd.DataFrame): The CAD data to be merged.
    process_data (pd.DataFrame): The process data to be merged.
    key_merge (str): The column name to join on.
    key_new (str): The column name to set as the new index.

    Returns:
    pd.DataFrame: The merged dataframe with `key_new` as the index.
    """
    # append process data with column key_merge from link_data
    process_data = process_data.join(link_data.set_index(key_new), on=key_new)
    # natural join data with cad_data on column key
    data = cad_data.join(process_data.set_index(key_merge), on=key_merge)
    # drop colum key_merge
    data.drop(columns=[key_merge], inplace=True)
    # set key_new
    data.set_index(key_new, inplace=True)
    return data


def preprocessing(data, num_columns, bin_columns, cat_columns):
    """
    Preprocesses the input data by scaling numerical columns and encoding categorical columns.

    Parameters:
    data (pd.DataFrame): The input dataframe containing the data to be preprocessed.
    num_columns (list of str): List of column names corresponding to numerical features.
    bin_columns (list of str): List of column names corresponding to binary features.
    cat_columns (list of str): List of column names corresponding to categorical features.

    Returns:
    pd.DataFrame: A dataframe with preprocessed data where numerical columns are scaled,
                  categorical columns are one-hot encoded, and binary columns are unchanged.
    """
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


def prepare_data(cad_data, process_data, link_data, num_columns, bin_columns, cat_columns, key_cad, key_process):
    """
    Prepares the data by selecting, merging, and preprocessing it.

    This function performs the following steps:
    1. Selects the relevant data from the provided datasets.
    2. Merges the selected data into a single dataset.
    3. Preprocesses the merged data by handling numerical, binary, and categorical columns.

    Args:
        cad_data (pd.DataFrame): The CAD data to be processed.
        process_data (pd.DataFrame): The process data to be processed.
        link_data (pd.DataFrame): The linking data to be used for merging.
        key_cad (str): The key column in the CAD data for merging.
        key_process (str): The key column in the process data for merging.
        num_columns (list of str): List of numerical columns to be preprocessed.
        bin_columns (list of str): List of binary columns to be preprocessed.
        cat_columns (list of str): List of categorical columns to be preprocessed.

    Returns:
        tuple: A tuple containing:
            - data (pd.DataFrame): The merged data.
            - data_preprocessed (pd.DataFrame): The preprocessed data.
    """
    ### calls select_data, merge_data and preprocessing functions
    # select the data
    cad_data, process_data, link_data = select_data(cad_data, process_data, link_data, key_cad, key_process)
    # merge the data
    data = merge_data(cad_data, process_data, link_data, key_merge=key_cad, key_new=key_process)
    # preprocess the data
    data_preprocessed = preprocessing(data, num_columns, bin_columns, cat_columns)
    return data, data_preprocessed
