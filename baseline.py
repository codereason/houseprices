import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def read_data():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    return train_df, test_df


def feature_engineering(df):

    df = df.drop('id')
    df = pd.get_dummies(df)
    df = df.apply(lambda x: x.fillna(x.median()),axis=0)
    return df






if __name__ == "__main__":

    train_df,test_df = read_data()

    train_df = feature_engineering(train_df)
    test_df = feature_engineering(test_df)



    my_submission = pd.DataFrame({'Id': test_df.Id, 'SalePrice': Y_pred})
    my_submission.to_csv('to_submit_%d.csv' % (1), index=False)



