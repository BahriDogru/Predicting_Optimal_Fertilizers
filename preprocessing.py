import pandas as pd
import numpy as np
import joblib

def data_preprocessing(data):
    print("############ Outlier starting ################")
    ################### outlier ##################
    numerical_col = [col for col in data.columns if data[col].dtype == "int64"]

    def outliers_thresholds(dataframe, variable, q1=0.10, q3=0.90):
        quartile1 = dataframe[variable].quantile(q1)
        quartile3 = dataframe[variable].quantile(q3)
        iqr = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * iqr
        low_limit = quartile1 - 1.5 * iqr
        return low_limit, up_limit

    def replace_with_threshold(dataframe, variable):
        low_limit, up_limit = outliers_thresholds(dataframe, variable)
        dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
        dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

    def check_outliers(dataframe, variable):
        low_limit, up_limit = outliers_thresholds(dataframe, variable)
        if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
            return True
        else:
            return False

    print('--------------------- Ceck Outlier Columns --------------------')
    for col in numerical_col:
        print(col, check_outliers(data, col))

    for col in numerical_col:
        if check_outliers(data, col):
            replace_with_threshold(data,col)
    print('---------------- Edited Outliers Columns -----------------------')
    for col in numerical_col:
        print(col, check_outliers(data, col))
    print("############ Outlier End ################")

    print("############ Feature Engineering starting ################")
    ################### Feature Engineering ##########################
    data['New_Temparature_Cat'] = pd.qcut(data['Temparature'], q=3, labels=["cold", "mean", "hot"]).astype(
        'object')

    data['New_Humidity_Cat'] = pd.qcut(data['Humidity'], q=3, labels=["dry", "normal", "humid"]).astype(
        'object')

    cat_col = ['Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    for col in cat_col:
        data[f'New_{col}_Cat'] = pd.qcut(data[col], q=3, labels=["low", "medium", "high"]).astype('object')

    print("############ Feature Engineering End ################")

    print("############ Encoding starting ################")
    ####################### Encoding ######################################################
    categorical_columns = [col for col in data.columns if data[col].dtype == 'O']

    loaded_ohe = joblib.load('onehot_encoder.pkl')
    ohe_array = loaded_ohe.transform(data[categorical_columns])

    encoded_cols = loaded_ohe.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(ohe_array, columns=encoded_cols, index=data.index)

    data.drop(categorical_columns, axis=1, inplace=True)
    data = pd.concat([data, encoded_df], axis=1)

    return data
