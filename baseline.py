import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import  xgboost as xgb
import lightgbm as lgb


def read_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df


def handle_feature(df):
    df = pd.get_dummies(df)
    print(' Features shape: ', df.shape)
    df = df.apply(lambda x: x.fillna(x.median()),axis=0)
    return df





train_df = pd.read_csv('train.csv')
all_data = pd.read_csv('features.csv')
all_data = all_data.iloc[:,3:]

print('all_data.describe()',all_data.describe())

ntrain = train_df.shape[0]

target = train_df.pop('SalePrice')



x_train = all_data[:ntrain]

x_test = all_data[ntrain:]



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,
                             learning_rate=0.05, max_depth=3,
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                              nthread = -1)

model_xgb.fit(x_train, target)
model_xgb_res = model_xgb.predict(x_test)
submission = pd.read_csv("sample_submission.csv")
submission['SalePrice'] = model_xgb_res

submission.to_csv('to_submit_20:19-2018-8-12.csv',index=False)

# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': Y_pred_1})
# my_submission.to_csv('to_submit_%s.csv' % ('lgb'), index=False)



