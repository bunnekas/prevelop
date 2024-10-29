### Prevelop
### Author: Kaspar Bunne

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler


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
    if 'Benennung (CAD)' in data.columns:
        data = data[['Benennung (CAD)', 'Klasse', 'Volumen [mm3]', 'Masse [kg]',
                'Flächeninhalt [mm2]', 'L [mm]', 'B [mm]', 'H [mm]', 'Lrot [mm]',
                'Da max. [mm]', 'Di min. [mm]']]
        data = data.rename(columns={"Benennung (CAD)": "Zeichnung"})
        data = data.dropna(subset=["Zeichnung"])
    if 'Key' in data.columns:
        data = data.rename(columns={"Key": "Zeichnung"})
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

#test