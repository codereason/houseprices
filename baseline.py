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
from models import eval_model,rmse,StackingAveragedModels
import os
from scipy.optimize import minimize



def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    print('='*30)
    print('=' * 30)
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test_df['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)


def rmsle_cv(model):
    n_folds=5
    kf = KFold(n_folds, shuffle=True, random_state=42)
    rmse= cross_val_score(model, x_train.values, target.values, cv = kf.get_n_splits())
    return(rmse)

train_df = pd.read_csv('train.csv')
ntrain = train_df.shape[0]
target = train_df.pop('SalePrice')
target = np.log1p(target)   #target log
test_df = pd.read_csv('test.csv')



all_data = pd.read_csv('features.csv')
all_data = all_data.iloc[:,3:]
all_data = np.log1p(all_data)


x_train = all_data[:ntrain]
x_test = all_data[ntrain:]



SEED = 42

xgb1 = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,  # 0.01
                 max_depth=4,  # 4
                 min_child_weight=1.5,
                 n_estimators=7200,  # 3000
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 random_state=SEED,
                 silent=1)

best_alpha = 0.00099
lasso = Lasso(alpha=best_alpha, max_iter=50000)

enet = ElasticNet(alpha=0.001)

kr = KernelRidge(alpha=0.3, kernel='polynomial', degree=2, coef0=1.85)

# regr5 = svm.SVR(kernel='rbf')

kernel = 1.0**2 * Matern(length_scale=1.0,
                         length_scale_bounds=(1e-05, 100000.0), nu=0.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=5e-9,
                                 optimizer='fmin_l_bfgs_b',
                                 n_restarts_optimizer=0,
                                 normalize_y=False,
                                 copy_X_train=True,
                                 random_state=SEED)

rfr = RandomForestRegressor(n_estimators=200, max_features='auto',
                                max_depth=12, min_samples_leaf=2)

lgb = lightgbm.LGBMRegressor()
# eval_model(regr1, x_train, target)
# # eval_model(model_lasso, x_train, target)
# eval_model(regr2, x_train,target)
# # eval_model(model_EN, x_train,target)
# eval_model(regr3, x_train,target)
# eval_model(regr5,x_train,target)
# eval_model(en_regr,x_train,target)
#
# eval_model(lgb_regr,x_train,target)



#预测出来的的结果都需要做expm1才能提交
# 做验证的时候就不需要 因为rmse的输入是(log(1+y_pred),log(1+ground_truth))
# regr1.fit(x_train, target)
# model_xgb_res = np.expm1(regr1.predict(x_test))
#
# create_submission(model_xgb_res,0)




def weight_ens():
    '''
    doing ensemble with weights instead of just averaging them
    use some basic regressor achieve 0.129
    '''
    split = int(0.8 * ntrain)

    regrs = [ lasso,enet,kr,gp,lgb,xgb1,rfr ]
    predictions = []
    for regr in regrs:

        predictions.append(regr.fit(x_train[:split], target[:split]).predict(x_train[split:]))

    def ensemble_rmse(weights):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction

        return rmse(target[split:], final_prediction)

    starting_values = [0.5]*len(predictions)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    # our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(predictions)

    res = minimize(ensemble_rmse, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    submissions = []
    for regr in regrs:
        submissions.append(regr.predict(x_test))
    final_submission  = 0
    for weight ,submission in zip(res['x'],submissions):
        final_submission += weight*submission
    return final_submission,res['x'],res['fun']

def weight_ens2():
    '''
    doing ensemble with weights instead of just averaging them
    use some basic regressor achieve 0.129
    '''
    split = int(0.8 * ntrain)

    regrs = [ lgb,xgb1,rfr  ]
    predictions = []
    for regr in regrs:

        predictions.append(regr.fit(x_train[:split], target[:split]).predict(x_train[split:]))

    def ensemble_rmse(weights):
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction

        return rmse(target[split:], final_prediction)

    starting_values = [0.5]*len(predictions)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    # our weights are bound between 0 and 1
    bounds = [(0, 1)] * len(predictions)

    res = minimize(ensemble_rmse, starting_values, method='SLSQP', bounds=bounds, constraints=cons)

    print('Ensamble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))
    submissions = []
    for regr in regrs:
        submissions.append(regr.predict(x_test))
    final_submission  = 0
    for weight ,submission in zip(res['x'],submissions):
        final_submission += weight*submission
    return final_submission,res['x'],res['fun']


# final_submission_1,weights,scores = weight_ens()
# final_submission_2,weights2,scores2 = weight_ens2()
# final_submission = 0.5 * final_submission_1 + 0.5*final_submission_2


stacked_averaged_models  = StackingAveragedModels(base_models=(enet, gp, kr),meta_model=lasso)

final_submission = np.expm1(final_submission)
create_submission(final_submission,scores=None)
