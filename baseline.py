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



def read_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df


def handle_feature(df):
    df = pd.get_dummies(df)
    print(' Features shape: ', df.shape)
    df = df.apply(lambda x: x.fillna(x.median()),axis=0)
    return df






if __name__ == "__main__":

    train_df,test_df = read_data()

    train_df = handle_feature(train_df)
    test_df = handle_feature(test_df)
    train_labels = train_df['SalePrice']
    train, test = train_df.align(test_df, join='inner', axis=1)
    X_train = train
    Y_train = train_labels
    X_test = test




    SEED = 2018
    regr1 = xgb.XGBRegressor(
        colsample_bytree=0.2,
        gamma=0.0,
        learning_rate=0.01,  # 0.01
        max_depth=4,  # 4
        min_child_weight=2,
        n_estimators=7200,  # 3000
        reg_alpha=0.9,
        reg_lambda=0.6,
        subsample=0.2,
        random_state=SEED,
        silent=True)
    mod = regr1.fit(X_train, Y_train)
    Y_pred = mod.predict(X_test)

    my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': Y_pred})
    my_submission.to_csv('to_submit_%d.csv' % (1), index=False)



