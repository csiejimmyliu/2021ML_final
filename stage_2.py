#%%
# Import librarie
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import metrics   #Additional scklearn functions

from imblearn.over_sampling import RandomOverSampler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 12, 4

APPLY_NORMALIZATION = True

#%%
if (APPLY_NORMALIZATION):
    train = pd.read_csv('./preprocessed_data/train_data_normalized.csv')
    train = train.loc[train['Churn Category'] != -1]
    test_data = pd.read_csv('./preprocessed_data/test_data_normalized.csv')
else:
    train = pd.read_csv('./preprocessed_data/train_data.csv')
    train = train.loc[train['Churn Category'] != -1]
    test_data = pd.read_csv('./preprocessed_data/test_data.csv')

target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]
NUM_CLASS = len(train[target].unique())


#%%
train_1to5 = train.loc[train['Churn Category']  != 0]
train_1to5.loc[train_1to5['Churn Category'] == 5, 'Churn Category'] = 0
train_1to5  

# %%
xgb1 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)

# %%
oversample = RandomOverSampler()
X = train_1to5.iloc[:, 1:77]
y = train_1to5.iloc[:, 77]
X_res, y_res = oversample.fit_resample(X, y)

# %%
y_res.value_counts()

# %%
cv_results = cross_validate(xgb1, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']

# %%
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
gsearch1 = GridSearchCV(estimator = xgb1, param_grid = param_test1, scoring='accuracy',n_jobs=4, cv=5)
gsearch1.fit(X_res,y_res)

# %%
gsearch1.best_params_, gsearch1.best_score_, gsearch1.cv_results_
# %%
xgb2 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=9,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
cv_results = cross_validate(xgb2, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']
# %%
param_test2 = {
    'max_depth':range(9,13),
    'gamma':[i/10.0 for i in range(0,5)]
}
gsearch2 = GridSearchCV(estimator = xgb2, param_grid = param_test2, scoring='accuracy',n_jobs=8, cv=5)
gsearch2.fit(X_res,y_res)
# %%
gsearch2.best_params_, gsearch2.best_score_, gsearch2.cv_results_
# %%
xgb3 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=11,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
cv_results = cross_validate(xgb3, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']
# %%
param_test3 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch3 = GridSearchCV(estimator = xgb3, param_grid = param_test3, scoring='accuracy',n_jobs=16, cv=5)
gsearch3.fit(X_res,y_res)
gsearch3.best_params_, gsearch3.best_score_, gsearch3.cv_results_

# %%
xgb4 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=11,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    objective= 'multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
cv_results = cross_validate(xgb4, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']

# %%
param_test4 = {
    'subsample':[i/100.0 for i in range(75,90,5)],
    'colsample_bytree':[i/100.0 for i in range(55,70,5)]
}
gsearch4 = GridSearchCV(estimator = xgb4, param_grid = param_test4, scoring='accuracy',n_jobs=16, cv=5)
gsearch4.fit(X_res,y_res)
gsearch4.best_params_, gsearch4.best_score_, gsearch4.cv_results_

# %%
xgb5 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=11,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    objective= 'multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
cv_results = cross_validate(xgb5, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']

# %%
param_test5 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
gsearch5 = GridSearchCV(estimator = xgb5, param_grid = param_test5, scoring='accuracy',n_jobs=16, cv=5)
gsearch5.fit(X_res,y_res)
gsearch5.best_params_, gsearch5.best_score_, gsearch5.cv_results_

# %%
xgb6 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=11,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.01,
    objective= 'multi:softprob',
    nthread=16,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
cv_results = cross_validate(xgb6, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']

# %%
param_test6 = {
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05]
}
gsearch6 = GridSearchCV(estimator = xgb6, param_grid = param_test6, scoring='accuracy',n_jobs=16, cv=5)
gsearch6.fit(X_res,y_res)
gsearch6.best_params_, gsearch6.best_score_, gsearch6.cv_results_

# %%
xgb7 = XGBClassifier(
    num_class=5,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=11,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0.01,
    objective= 'multi:softprob',
    nthread=16,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
cv_results = cross_validate(xgb7, X_res, y_res, cv=5, scoring='accuracy')
cv_results['test_score']