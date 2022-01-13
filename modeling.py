#%%
# Import librarie
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics   #Additional scklearn functions

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

APPLY_NORMALIZATION = True

#%%
if (APPLY_NORMALIZATION):
    train_data = pd.read_csv('./preprocessed_data/train_data_normalized.csv')
    train_data = train_data.loc[train_data['Churn Category'] != -1]
    target = 'Churn Category'
    IDcol = 'Customer ID'
    test_data = pd.read_csv('./preprocessed_data/test_data_normalized.csv')
else:
    train_data = pd.read_csv('./preprocessed_data/train_data.csv')
    train_data = train_data.loc[train_data['Churn Category'] != -1]
    target = 'Churn Category'
    IDcol = 'Customer ID'
    test_data = pd.read_csv('./preprocessed_data/test_data.csv')

NUM_CLASS = len(train_data[target].unique())

#%%
def modelfit(alg, dtrain, predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='merror', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='merror')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

#%%
#Choose all predictors except target & IDcols
predictors = [x for x in train_data.columns if x not in [target, IDcol]]
xgb1 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=4,
    scale_pos_weight=1,
    num_class=NUM_CLASS,
    seed=1126
)
modelfit(xgb1, train_data, predictors)

#%%
xgb2 = XGBClassifier(
    learning_rate =0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=4,
    scale_pos_weight=1,
    num_class=NUM_CLASS,
    seed=1126
)
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(
    estimator=xgb2,param_grid=param_test1,scoring='accuracy',n_jobs=4,cv=5
)
gsearch1.fit(train_data[predictors],train_data[target])
gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_

# %%
gsearch1.cv_results_