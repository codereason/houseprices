import warnings
warnings.simplefilter("ignore", UserWarning)
import datetime

import lightgbm
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, Matern, RBF, DotProduct, RationalQuadratic
from models import AverageEnsemble, StackingEnsemble, eval_model
from models import eval_model


def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test_df['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


def rmsle_cv(model):
    n_folds=5
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= cross_val_score(model, x_train.values, target.values, cv = kf.get_n_splits())
    return(rmse)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
all_data = pd.read_csv('features.csv')
all_data = all_data.iloc[:,3:]
all_data = np.log1p(all_data)


ntrain = train_df.shape[0]

target = train_df.pop('SalePrice')
target = np.log1p(target)

x_train = all_data[:ntrain]
x_test = all_data[ntrain:]



SEED = 2018
model_xgb = xgb.XGBRegressor()
model_lgb=lightgbm.LGBMRegressor()
# model_lasso = Lasso()
model_ridge = Ridge()
# model_EN = ElasticNet()
model_KR = KernelRidge()
model_rfr = RandomForestRegressor()
model_etr = ExtraTreesRegressor()

eval_model(model_xgb, x_train, target)
eval_model(model_lgb, x_train, target)
# eval_model(model_lasso, x_train, target)
eval_model(model_ridge, x_train,target)
# eval_model(model_EN, x_train,target)
eval_model(model_KR, x_train,target)
eval_model(model_rfr,x_train,target)


# model_xgb.fit(x_train, target)
# model_xgb_res = model_xgb.predict(x_test)
#
# create_submission(model_xgb_res,0)
# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': Y_pred_1})
# my_submission.to_csv('to_submit_%s.csv' % ('lgb'), index=False)



