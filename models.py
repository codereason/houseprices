from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import numpy as np

def rmse(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))



def eval_model(model,X,y,n_splits=5):

    score = np.zeros(n_splits)
    i = 0
    kf = KFold(n_splits=n_splits, random_state=50)
    for train_ind,test_ind in kf.split(X):

        xtrain,ytrain = X.iloc[train_ind],y.iloc[train_ind]
        # print('xtrain shape is ',xtrain.describe(),'ytrain shape is ',ytrain.describe())
        model.fit(xtrain,ytrain)
        y_cv = model.predict(X.iloc[test_ind])
        score[i] = rmse(y_cv,y.iloc[test_ind])
        i+=1
    mean_rmse = np.mean(score)
    print('%s :'% model,'mean RMSE is %.5f'%mean_rmse,'std RMSE is ', np.std(score))

    return mean_rmse





class AverageEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, regressors=None):
        self.regressors = regressors

    def fit(self, X, y):
        for regressor in self.regressors:
            regressor.fit(X, y)

    def predict(self, X):
        self.predictions_ = list()
        for regressor in self.regressors:
            self.predictions_.append(regressor.predict(X).ravel())

        # res = 0.45*self.predictions_[1] + 0.25*self.predictions_[0] + 0.30*self.predictions_[2]
        res = np.mean(self.predictions_, axis=0)

        return res

class StackingEnsemble(object):
    def __init__(self, n_splits, stacker, base_models):
        self.n_splits = n_splits
        self.stacker = stacker
        self.base_models = base_models

    def fit_predict(self, X, y, T):
        X = np.array(X)
        y = np.array(y)
        T = np.array(T)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)

        S_train = np.zeros((X.shape[0], len(self.base_models)))
        S_test = np.zeros((T.shape[0], len(self.base_models)))
        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((T.shape[0], kf.get_n_splits()))
            for j, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                # y_holdout = y[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout).ravel()
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = clf.predict(T).ravel()
            S_test[:, i] = S_test_i.mean(1)
        self.stacker.fit(S_train, y)
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred