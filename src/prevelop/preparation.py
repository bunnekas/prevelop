### PrEvelOp
### Data loading and preprocessing module
### Author: Kaspar Bunne

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler


def load_data(file):
    """
    Load data from a file and return it as a pandas DataFrame.
    Supports CSV and Excel formats.

    Parameters:
    file (str): Path to the file. Supports '.csv' and '.xlsx' extensions.

    Returns:
    pd.DataFrame: The loaded data.

    Raises:
    ValueError: If the file format is not supported.
    """
    if file.endswith('.csv'):
        return pd.read_csv(file)
    elif file.endswith('.xlsx'):
        return pd.read_excel(file)
    else:
        raise ValueError(f'File format not supported: {file}')


def aggregate_data(data, key, columns, methods):
    """
    Aggregate data based on specified methods for each column.

    Parameters:
    data (pd.DataFrame): Input data to be aggregated.
    key (str): Column name to group by.
    columns (list of str): Columns to aggregate.
    methods (list of str): Aggregation method per column.
        Supported: 'encode', 'sum', 'mean', 'max', 'min'.

    Returns:
    tuple:
        - data_new (pd.DataFrame): Aggregated data with key as index.
        - num_columns (list of str): Numerical column names.
        - cat_columns (list of str): Categorical column names.
    """
    keys = list(set(data[key].tolist()))
    data_new = pd.DataFrame(data=keys, columns=[key])
    cat_columns = []
    num_columns = []

    for column, method in zip(columns, methods):
        if method == 'encode':
            ### one-hot encode categorical column across grouped rows
            column_list = []
            for k in keys:
                k_data = data[data[key] == k]
                column_list.append(k_data[column].tolist())
            data_new[column] = column_list
            # get unique values excluding nan
            values = list(set(
                item for sublist in column_list
                for item in sublist if str(item) != 'nan'
            ))
            for value in values:
                col_name = f'{column}_{value}'
                data_new[col_name] = [
                    1 if value in items else 0
                    for items in data_new[column]
                ]
                cat_columns.append(col_name)
            data_new.drop(columns=[column], inplace=True)
        elif method in ('sum', 'mean', 'max', 'min'):
            data_new[column] = getattr(data.groupby(key)[column], method)().tolist()
            num_columns.append(column)
        else:
            raise ValueError(f'Aggregation method not supported: {method}')

    data_new.set_index(key, inplace=True)
    return data_new, num_columns, cat_columns


def preprocessing(data, num_columns, cat_columns):
    """
    Preprocess data by scaling numerical columns and encoding categorical columns.

    Numerical columns are scaled using MaxAbsScaler. Categorical columns are
    one-hot encoded using OneHotEncoder.

    Parameters:
    data (pd.DataFrame): Input data.
    num_columns (list of str): Numerical feature column names.
    cat_columns (list of str): Categorical feature column names.

    Returns:
    pd.DataFrame: Preprocessed data with scaled numerical and encoded categorical features.
    """
    num_columns = [col for col in num_columns if col in data.columns]
    cat_columns = [col for col in cat_columns if col in data.columns]

    ### scale numerical columns with MaxAbsScaler
    df_num = data[num_columns]
    scaler = MaxAbsScaler().fit(df_num)
    df_num_scaled = pd.DataFrame(
        data=scaler.transform(df_num),
        index=df_num.index,
        columns=df_num.columns,
    )

    ### encode categorical columns with OneHotEncoder
    df_cat = data[cat_columns]
    encoder = OneHotEncoder(sparse_output=False).fit(df_cat)
    df_cat_encoded = pd.DataFrame(
        data=encoder.transform(df_cat),
        index=df_cat.index,
        columns=encoder.get_feature_names_out(cat_columns),
    )

    ### concatenate scaled and encoded features
    data_preprocessed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    return data_preprocessed


def prepare_data(data, num_columns, cat_columns):
    """
    Prepare data for clustering by cleaning and preprocessing.

    Handles missing values, removes sparse columns, and applies
    scaling and encoding via the preprocessing function.

    Parameters:
    data (pd.DataFrame): Raw input data.
    num_columns (list of str): Numerical feature column names.
    cat_columns (list of str): Categorical feature column names.

    Returns:
    tuple:
        - data (pd.DataFrame): Cleaned data.
        - data_preprocessed (pd.DataFrame): Preprocessed data ready for clustering.
    """
    ### remove sparse columns (more than 95% identical values)
    data = data.loc[:, (data != 0).mean() > 0.05]
    data = data.loc[:, (data != 1).mean() > 0.05]

    ### remove columns with problematic names
    data = data.loc[:, ~data.columns.str.contains('nan')]
    data = data.loc[:, ~data.columns.str.contains('Unnamed')]

    ### drop rows with missing values
    data = data.dropna()

    ### update column lists based on remaining columns
    num_columns = [col for col in num_columns if col in data.columns]
    cat_columns = [col for col in cat_columns if col in data.columns]

    ### preprocess
    data_preprocessed = preprocessing(data, num_columns, cat_columns)
    return data, data_preprocessed
