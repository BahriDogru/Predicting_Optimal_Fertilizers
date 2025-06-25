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
import joblib
import preprocessing


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
joblib.dump(ohe, 'onehot_encoder.pkl')
train_df.drop(categorical_columns,axis=1, inplace=True)
train_df = pd.concat([train_df,encoded_df], axis=1)


le = LabelEncoder()
train_df['Fertilizer Name'] =  le.fit_transform(train_df['Fertilizer Name'])
joblib.dump(le, 'fertilizer_label_encoder.pkl')
#################################################
# Modeling
#################################################

######## Base Model selection ##################
X = train_df.drop('Fertilizer Name', axis=1)
y = train_df['Fertilizer Name']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=7)


models = {
    'XGBoost': XGBClassifier(random_state=7, verbosity=0),
    'CatBoost': CatBoostClassifier(random_state=7, verbose=False),
    'LightGBM':LGBMClassifier(random_state=7,verbose=-1)
}

for name, model in models.items():
    print(f'------------{name}----------------')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(classification_report(y_val, y_pred))
    print('############################################')


######################## Base model selection with MAP@K algorithm ##############################
def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true_label, pred_labels in zip(y_true, y_pred):
        try:
            rank = pred_labels.index(true_label)
            if rank < k:
                score += 1.0 / (rank + 1)
        except ValueError:
            continue  # true_label not in top-k predictions
    return score / len(y_true)


for name, model in models.items():
    print(f"------------- {name} -------------")
    model.fit(X_train, y_train)

    # Olasılık tahminleri
    y_proba = model.predict_proba(X_val)

    # İlk 3 tahmini al
    # Her satırda (yani her örnekte) sınıfların olasılıklarını küçükten büyüğe sıralar.
    # Ama değerleri değil, bu değerlerin index’lerini döner.
    # [:, -3:] Son 3 elemanı alır → yani en yüksek 3 olasılığa sahip sınıfların index’leri.
    top_3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]

    # MAP@3 hesapla
    score = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)

    print(f"MAP@3 score: {score:.4f}")
    print("###########################################")

# Şimdi bu indexleri tekrar string sınıf isimlerine çeviriyoruz
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

############################# Hiperparameter Optimization ###################################

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 16),
        'n_estimators': trial.suggest_int('n_estimators', 900, 5000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.09, log=True),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0.09, 0.7),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.05, 0.5),
        'max_delta_step': trial.suggest_int('max_delta_step', 2, 9),
        'objective': 'multi:softprob',
        'random_state': 7,
        'n_jobs': -1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss',
        'device': 'cuda',
    }

    xgb_model = XGBClassifier(**params)

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=False
    )

    y_proba = xgb_model.predict_proba(X_val)
    top_3_preds = np.argsort(y_proba, axis=1)[:, -3:][:, ::-1]
    score = mapk(y_val.tolist(), top_3_preds.tolist(), k=3)

    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)


############################ Final Model ############################################

best_params = {
        'max_depth':11,
        'colsample_bytree':0.3508548763606836,
        'subsample':0.6142056550341887,
        'n_estimators':3413,
        'learning_rate':0.009318260040590403,
        'gamma':0.452760148679334,
        'max_delta_step':6,
        'reg_alpha':0.13339412011885346,
        'reg_lambda':0.46614123300729193,
        'objective':'multi:softprob',
        'random_state':7,
        'enable_categorical':True,
        'n_jobs':-1,
        'eval_metric':'mlogloss'
}

final_model = XGBClassifier(**best_params)
final_model.fit(X, y)



test_id = test_df['id']
test_df.drop('id', axis=1, inplace=True)

test_df = preprocessing.data_preprocessing(test_df)
test_proba = final_model.predict_proba(test_df)
top_3_preds = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)



submission = pd.DataFrame({
   'id': test_id,  # test setindeki örnek id'leri
   'predictions': [' '.join(row) for row in top_3_labels]
})

submission.columns = ['id','Fertilizer Name' ]
submission.to_csv('submission2.csv', index=False)