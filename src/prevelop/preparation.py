### Prevelop
### preprocessing module for data loading and preprocessing
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
import warnings
warnings.filterwarnings("ignore")


def load_data(file):
    """
    Load data from a file and return it as a pandas DataFrame.
    This function supports loading data from CSV and Excel files. If the file
    format is not supported, an error message will be printed.
    Args:
        file (str): The path to the file to be loaded. The file should have
                    a '.csv' or '.xlsx' extension.
    Returns:
        pandas.DataFrame: The loaded data as a DataFrame if the file format
                          is supported.
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


def load_simus_data(file):
    """
    Load and preprocess simulation data from a CSV file.
    This function reads a CSV file containing simulation data, processes it to handle
    numerical and categorical columns, and returns a cleaned and structured DataFrame
    along with the processed column names.
    Args:
        file (str): The file path to the CSV file to be loaded.
    Returns:
        tuple:
            - cad_data (pd.DataFrame): A DataFrame containing the processed simulation data,
              with numerical and categorical columns combined.
            - columns_num (list of str): A list of numerical column names that were successfully
              processed and included in the output.
            - columns_cat (list of str): A list of categorical column names that were successfully
              processed and included in the output.
    Notes:
        - The function attempts to read the CSV file using multiple encodings (UTF-8, Latin-1, ISO-8859-1).
        - Duplicate rows are removed based on the 'OBJECTKEY', 'NAME', and 'VALUE' columns.
        - Numerical data is pivoted, cleaned, and converted to float type.
        - Categorical data is expanded into binary columns for each unique value, excluding
          non-meaningful values such as 'Unbekannt', '0', and 'invalid'.
        - The resulting DataFrame uses the 'Zeichnung' column as the index.
    """
    # Specify the numerical and categorical columns
    num_columns = ['Volumen','L','B','H','Da max.','Di min.','Lrot','Gesamtanzahl Bohrungen','Anzahl Außenabsätze', 
               'Gesamtanzahl Eindrehungen außen','Gesamtanzahl Eindrehungen innen','Fasenbreite rechts', 
               'Endenwinkel rechts','Anzahl Innenabsätze']
    cat_columns = ['Klasse','Eindrehungsart außen','Eindrehungsanordnung außen','Absatzform','Bohrungsanordnung',
               'Anbringung Bohrungsanordnung','Bohrungsart','Ende rechts','Innenform']
    
    # Read the CSV file with the correct encoding
    try:
        # Try reading with UTF-8 first
        df = pd.read_csv(file, delimiter=';', encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try other common encodings
        try:
            df = pd.read_csv(file, delimiter=';', encoding='latin1')
        except UnicodeDecodeError:
            df = pd.read_csv(file, delimiter=';', encoding='ISO-8859-1')

    # Drop duplicate rows based on 'OBJECTKEY' and 'NAME', keeping the first occurrence
    df_unique = df.drop_duplicates(subset=['OBJECTKEY', 'NAME', 'VALUE'], keep=False)

    columns = df_unique['NAME'].unique()
    # remove columns from columns_num that are not in columns
    columns_num = [col for col in num_columns if col in columns]
    # remove columns from columns_cat that are not in columns
    columns_cat = [col for col in cat_columns if col in columns]

    # select rows with 'NAME' in columns_num
    df_num = df_unique[df_unique['NAME'].isin(columns_num)]
    # select rows with 'NAME' in columns_cat
    df_cat = df_unique[df_unique['NAME'].isin(columns_cat)]
    # make all values in df_bin categorical
    df_cat['VALUE'] = df_cat['VALUE'].astype('category')

    ### numerical data
    # Pivot the table
    pivot_df_num = df_num.pivot(index='OBJECTKEY', columns='NAME', values='VALUE')
    # remove the column name
    pivot_df_num.columns.name = None
    # Create a column 'Zeichnung' on the left side of the pivot table
    pivot_df_num.insert(0, 'Zeichnung', pivot_df_num.index)
    # remove the indices
    pivot_df_num.reset_index(drop=True, inplace=True)
    # select specified columns
    data_num = pivot_df_num[columns_num]
    # clean data, replace ',' with '.'
    data_num = data_num.apply(lambda x: x.str.replace(',', '.'))
    # fill NaN values with 0
    data_num.fillna(0, inplace=True)
    # make data type float
    data_num = data_num.astype(float)
    # add the column 'Zeichnung' to the dataframe
    data_num['Zeichnung'] = pivot_df_num['Zeichnung']

    ### categorical data
    # create empty dataframe with column Zeichnung
    data_cat = pd.DataFrame(columns=['Zeichnung'], data=data_num['Zeichnung'])
    # iterate over the binary columns and create a new column for each value
    for col in columns_cat:
        # create a list of unique values in the column
        unique_values = df_cat['VALUE'].unique()
        # remove non meaningful values
        unique_values = [x for x in unique_values if str(x) != 'nan']
        if 'Unbekannt' in unique_values:
            unique_values.remove('Unbekannt')
        if 'unbekannt' in unique_values:    
            unique_values.remove('unbekannt')
        if '0' in unique_values:
            unique_values.remove('0')
        if 'invalid' in unique_values:
            unique_values.remove('invalid')
        if 'Invalid' in unique_values:
            unique_values.remove('Invalid')
        if 'INVALID' in unique_values:
            unique_values.remove('INVALID')
        # create a new column for each value with column name 'col' + '_' + 'value'
        for value in unique_values:
            value = str(value)
            data_cat[col + '_' + value] = 0
            # iterate over the Zeichnung column
            for index, row in df_cat.iterrows():
                # if the value in the column is equal to the value in the row, set the value in the new column to 1
                if row['VALUE'] == value:
                    data_cat.loc[data_cat['Zeichnung'] == row['OBJECTKEY'], col + '_' + value] = 1
    columns_cat = list(data_cat.columns[1:])

    # merge the dataframes
    cad_data = pd.merge(data_num, data_cat, on='Zeichnung')
    
    # set the column 'Zeichnung' as index
    cad_data.set_index('Zeichnung', inplace=True)

    return cad_data, columns_num, columns_cat


# def clear_cad_data(data):
#     """
#     Cleans and preprocesses CAD data by performing the following operations:
    
#     1. Drops duplicate rows.
#     2. Drops rows with missing values in specific columns.
#     3. Renames columns:
#        - Renames 'Benennung (CAD)' to 'Zeichnung'.
#        - Renames 'Key' to 'Zeichnung' if 'Benennung (CAD)' is not present.
#     4. Deletes specific substrings ('mm3', 'kg', 'mm2', 'mm', ' ') from certain columns and converts them to float.
#     5. Drops rows with missing values in 'L [mm]', 'B [mm]', or 'H [mm]'.
    
#     Parameters:
#     data (pandas.DataFrame): The input CAD data to be cleaned.
    
#     Returns:
#     pandas.DataFrame: The cleaned and preprocessed CAD data.
#     """
#     columns = data.columns
#     # set key
#     if 'Benennung (CAD)' in columns:
#         data = data.rename(columns={"Benennung (CAD)": "Zeichnung"})
#         data = data.dropna(subset=["Zeichnung"])
#     elif 'Key' in data.columns:
#         data = data.rename(columns={"Key": "Zeichnung"})
#         data = data.dropna(subset=["Zeichnung"])
#     else:
#         raise ValueError("No key found.")
#     # remove units
#     columns_clean = ['Volumen [mm3]','Masse [kg]','Flächeninhalt [mm2]','L [mm]','B [mm]', 'H [mm]','Lrot [mm]','Da max. [mm]','Di min. [mm]']
#     for col in columns_clean:
#         if col in data.columns:
#             # delete substings mm3, kg, mm2, mm from columns Volumen, Masse, Flächeninhalt, L, B, H, Lrot, Da max., Di min.
#             data[col] = data[col].str.replace('mm3', '')
#             data[col] = data[col].str.replace('kg', '')
#             data[col] = data[col].str.replace('mm2', '')
#             data[col] = data[col].str.replace('mm', '')
#             data[col] = data[col].str.replace(' ', '')
#             data[col] = data[col].str.replace(',', '.')
#             data[col] = data[col].astype(float)
#     # drop rows with missing values in columns L [mm], B [mm] or H [mm]
#     data.dropna(subset=['L [mm]', 'B [mm]', 'H [mm]'], inplace=True)
#     # fill missing values with 0
#     data.fillna(0, inplace=True)
#     return data 


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
    cat_columns = []
    num_columns = []
    for column, method in zip(columns, methods):
        column_list = []
        if method == 'encode':
            for teil in teile_list:
                teil_data = data[data[key] == teil]
                column_list.append(teil_data[column].tolist())
            data_new[column] = column_list
            # select all values in column unique
            values = list(set([item for sublist in column_list for item in sublist if str(item) != 'nan']))
            for value in values:
                column_values = [1 if value in process else 0 for process in data_new[column]]
                data_new[str(column) + ' ' + str(value)] = column_values
                cat_columns.append(str(column) + ' ' + str(value))
            data_new.drop(columns=[column], inplace=True)
        elif method == 'sum':
            data_new[column] = data.groupby(key)[column].sum().tolist()
            num_columns.append(column)
        elif method == 'mean':
            data_new[column] = data.groupby(key)[column].mean().tolist()
            num_columns.append(column)
        elif method == 'max':
            data_new[column] = data.groupby(key)[column].max().tolist()
            num_columns.append(column)
        elif method == 'min':
            data_new[column] = data.groupby(key)[column].min().tolist()
            num_columns.append(column)
        else:
            raise ValueError('Method not supported')
    # set key as index
    data_new.set_index(key, inplace=True)
    return data_new, num_columns, cat_columns


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
    # select rows from cad_data with indices in column key_cad in link_data
    cad_data = cad_data[cad_data.index.isin(link_data[key_cad])]
    zeichnungen = cad_data.index.tolist()
    # select rows from process_data with process_data in column process_data in link_data
    process_data = process_data[process_data.index.isin(link_data[key_process])]
    teile = process_data.index.tolist()
    # select rows from link_data with key_cad in zeichnungen list 
    link_data = link_data[link_data[key_cad].isin(zeichnungen)]
    # select rows from data with key_process in teile list
    link_data = link_data[link_data[key_process].isin(teile)]
    # remove duplicates
    link_data = link_data.drop_duplicates()
    zeichnungen = link_data[key_cad].tolist()
    teile = link_data[key_process].tolist()
    cad_data = cad_data[cad_data.index.isin(zeichnungen)]
    process_data = process_data[process_data.index.isin(teile)]
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
    # index of cad_data is key_merge
    cad_data[key_merge] = cad_data.index
    # remove index 
    cad_data.reset_index(drop=True, inplace=True)
    # index of process_data is key_new
    process_data[key_new] = process_data.index
    # remove index
    process_data.reset_index(drop=True, inplace=True)
    # append process data with column key_merge from link_data
    process_data = process_data.join(link_data.set_index(key_new), on=key_new)
    # natural join data with cad_data on column key
    data = cad_data.join(process_data.set_index(key_merge), on=key_merge)
    # drop colum key_merge
    data.drop(columns=[key_merge], inplace=True)
    # set key_new
    data.set_index(key_new, inplace=True)
    return data


def preprocessing(data, num_columns, cat_columns):
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
    num_columns = [col for col in num_columns if col in data.columns]
    cat_columns = [col for col in cat_columns if col in data.columns]
    ### preprocess data: scale numerical columns, encode categorical columns
    # split the dataframe in with respect to the selected columns
    df_num = data[num_columns]
    df_cat = data[cat_columns]
    # scale the numerical columns with MaxAbsScaler
    scaler = MaxAbsScaler().fit(df_num)
    df_num_scaled = pd.DataFrame(data=scaler.transform(df_num), index=df_num.index, columns=df_num.columns)
    # encode the categorical columns with OneHotEncoder
    encoder = OneHotEncoder().fit(df_cat)
    df_cat_encoded = pd.DataFrame(data=encoder.transform(df_cat), index=df_cat.index, columns=encoder.get_feature_names_out(cat_columns))
    # remove the column name
    # concatenate the subdataframes columnwise
    data_preprocessed = pd.concat([df_num_scaled, df_cat_encoded], axis=1)
    return data_preprocessed


def prepare_data(cad_data, num_columns, cat_columns, process_data=None, link_data=None):
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
    if process_data is None:
        data = cad_data
        data_preprocessed = preprocessing(data, num_columns, cat_columns)  
        return data, data_preprocessed
    elif link_data is None:
        print('No link data provided')
        data = cad_data
        data_preprocessed = preprocessing(data, num_columns, cat_columns)  
        return data, data_preprocessed
    else:
        # key_cad is index of cad_data
        key_cad = cad_data.index.name
        # key_process is index of process_data
        key_process = process_data.index.name
        # select the data
        cad_data, process_data, link_data = select_data(cad_data, process_data, link_data, key_cad, key_process)
        # merge the data
        data = merge_data(cad_data, process_data, link_data, key_merge=key_cad, key_new=key_process)
        data[num_columns] = data[num_columns].astype(float)
        # data[cat_columns] = data[cat_columns].astype(int)
        data_preprocessed = preprocessing(data, num_columns, cat_columns)
        # remove columns in data and data_preprocessed containing the substring 'nan'
        data = data.loc[:, ~data.columns.str.contains('nan')]
        data_preprocessed = data_preprocessed.loc[:, ~data_preprocessed.columns.str.contains('nan')]
        # remove columns in data and data_preprocessed containing the substring 'Unnamed'
        data = data.loc[:, ~data.columns.str.contains('Unnamed')]
        data_preprocessed = data_preprocessed.loc[:, ~data_preprocessed.columns.str.contains('Unnamed')]
        # drop columns in data and data_preprocessed where more than 95% of the values are 0
        data = data.loc[:, (data != 0).mean() > 0.05]
        data_preprocessed = data_preprocessed.loc[:, (data_preprocessed != 0).mean() > 0.05]
        # drop columns in data and data_preprocessed where more than 95% of the values are 1
        data = data.loc[:, (data != 1).mean() > 0.05]
        data_preprocessed = data_preprocessed.loc[:, (data_preprocessed != 1).mean() > 0.05]
        # remove rows in data and data_preprocessed containing nan values
        data = data.dropna()
        data_preprocessed = data_preprocessed.dropna()
        return data, data_preprocessed
