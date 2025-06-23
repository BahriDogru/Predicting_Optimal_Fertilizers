import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import optuna

matplotlib.use('TkAgg')
pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)

# data
train_df = pd.read_csv('datasets/train.csv')
test_df = pd.read_csv('datasets/test.csv')

train_id = train_df['id']
train_df.drop('id', axis=1, inplace=True)

train_df.head()
test_df.head()

################################################
# Data Pre-Processing
#################################################

train_df.info()
test_df.info()

################### outlier ##################
numerical_col = [col for col in train_df.columns if train_df[col].dtype == "int64"]

for col in numerical_col:
    sns.boxplot(data=train_df, x=train_df[col])
    plt.title(f'{col} Distribution')
    plt.show()



def outliers_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    iqr = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * iqr
    low_limit = quartile1 - 1.5 * iqr
    return low_limit, up_limit

def check_outliers(dataframe, variable):
    low_limit, up_limit = outliers_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in numerical_col:
    print(col, check_outliers(train_df,col))


################### Feature Engineering ##########################
train_df['New_Temparature_Cat'] = pd.qcut(train_df['Temparature'],q=3,labels=["cold", "mean", "hot"]).astype('object')

train_df['New_Humidity_Cat'] = pd.qcut(train_df['Humidity'],q=3,labels=["dry", "normal", "humid"]).astype('object')


cat_col = ['Moisture','Nitrogen','Potassium','Phosphorous']
for col in cat_col:
    train_df[f'New_{col}_Cat'] = pd.qcut(train_df[col], q=3, labels=["low", "medium", "high"]).astype('object')


####################### Encoding ######################################################
categorical_columns = [col for col in train_df.columns if train_df[col].dtype == 'O']
categorical_columns.remove('Fertilizer Name')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe_array = ohe.fit_transform(train_df[categorical_columns])

encoded_cols = ohe.get_feature_names_out(categorical_columns)
encoded_df = pd.DataFrame(ohe_array, columns=encoded_cols, index=train_df.index)

train_df.drop(categorical_columns,axis=1, inplace=True)
train_df = pd.concat([train_df,encoded_df], axis=1)


le = LabelEncoder()
train_df['Fertilizer Name'] =  le.fit_transform(train_df['Fertilizer Name'])