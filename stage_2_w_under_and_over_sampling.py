#%%
# import library
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import StratifiedGroupKFold

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 12, 4

APPLY_NORMALIZATION = True
WITH_GROUPING = True
SEED = 326

#%%
# import data
if (APPLY_NORMALIZATION):
    train = pd.read_csv('./preprocessed_data/train_data_std_normalized.csv')
    train = train.loc[train['Churn Category'] != -1]
    test_data = pd.read_csv('./preprocessed_data/test_data_std_normalized.csv')
else:
    train = pd.read_csv('./preprocessed_data/train_data.csv')
    train = train.loc[train['Churn Category'] != -1]
    test_data = pd.read_csv('./preprocessed_data/test_data.csv')

target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]
NUM_CLASS = len(train[target].unique())


#%%
# drop category 0, category No. minus 1
train_1to5 = train.loc[train['Churn Category']  != 0].copy()
for i in range(1,6):
    train_1to5['Churn Category'].replace(i, i-1, inplace=True)
train_1to5['Churn Category'].value_counts()

#%%
# random undersampling
X = train_1to5[predictors]
y = train_1to5[target]
sampling_num = dict(y.value_counts())
sampling_num[0] = 300
underample = RandomUnderSampler(sampling_strategy=sampling_num, 
    random_state=SEED)
X_under, y_under = underample.fit_resample(X, y)
y_under.value_counts()

# %%
# random oversample and grouping
# prevent duplicate examples from appearing in both training and validation sets
if WITH_GROUPING:
    X_under['Group Label'] = np.array(list(range(len(X_under))))
oversample = RandomOverSampler(random_state=SEED)
X_res, y_res = oversample.fit_resample(X_under, y_under)
print(y_res.value_counts())
if WITH_GROUPING:
    groups = np.array(X_res['Group Label'])
    X_res.drop('Group Label', axis=1, inplace=True)
    group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)

#%%
# xgboost hyperparameters
TUNING_STAGES = 9
initial_params = {
    'num_class': 5,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 5,
    'min_child_weight': 1.0,
    'gamma': 0.0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'nthread': 4,
    'seed': SEED,
    'verbosity': 0,
    'use_label_encoder': False,
    'reg_alpha': 0
}
param_iterations = [initial_params] * (TUNING_STAGES + 1)
scores = [0] * (TUNING_STAGES + 1)

#%%
# cross validate model
def cv_model(params, data_X, data_y, folds):
    alg = XGBClassifier(
        num_class=params['num_class'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        objective=params['objective'],
        nthread=params['nthread'],
        seed=params['seed'],
        verbosity=params['verbosity'],
        use_label_encoder=params['use_label_encoder'],
        reg_alpha = params['reg_alpha']
    )
    cv_results = cross_validate(alg, data_X, data_y, 
        cv=folds.split(data_X, data_y, groups), scoring='f1_micro')
    return cv_results['test_score']

#%%
# graphing function
def graph_imp(params, data_X, data_y):
    alg = XGBClassifier(
        num_class=params['num_class'],
        learning_rate=params['learning_rate'],
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_child_weight=params['min_child_weight'],
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        objective=params['objective'],
        nthread=params['nthread'],
        seed=params['seed'],
        verbosity=params['verbosity'],
        use_label_encoder=params['use_label_encoder'],
        reg_alpha=params['reg_alpha']
    )
    alg.fit(data_X, data_y)
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

#%%
# grid search function
def grid_search(original_params, data_X, data_y, param_test, folds):
    alg = XGBClassifier(
        num_class=original_params['num_class'],
        learning_rate=original_params['learning_rate'],
        n_estimators=original_params['n_estimators'],
        max_depth=original_params['max_depth'],
        min_child_weight=original_params['min_child_weight'],
        gamma=original_params['gamma'],
        subsample=original_params['subsample'],
        colsample_bytree=original_params['colsample_bytree'],
        objective=original_params['objective'],
        nthread=original_params['nthread'],
        seed=original_params['seed'],
        verbosity=original_params['verbosity'],
        use_label_encoder=original_params['use_label_encoder'],
        reg_alpha=original_params['reg_alpha']
    )
    gsearch = GridSearchCV(estimator=alg, param_grid = param_test, 
        scoring='f1_micro',n_jobs=16, cv=folds.split(data_X, data_y, groups))
    gsearch.fit(data_X, data_y)
    return gsearch.best_score_, gsearch.best_params_


# %%
# initial model performance
iter_num = 0
scores[iter_num] = np.average(cv_model(param_iterations[iter_num], X_res, y_res, group_kfold))
print('cv performance:', scores[0])
print('initial params: ', param_iterations[iter_num])

#%%
# find initial n_estimators
xgtrain = xgb.DMatrix(X_res.values, y_res.values)
cvresult = xgb.cv(initial_params, xgtrain, num_boost_round=1000, 
    nfold=5, stratified=True,
    metrics='merror', early_stopping_rounds=50, verbose_eval=False)
print('initial n_estimaotrs: ', cvresult.shape[0])
param_iterations[0]['n_estimators'] = cvresult.shape[0]

# %%
# grid search 1
iter_num = 1
param_test1 = {
    'max_depth':range(3, 10, 2),
    'min_child_weight':range(1, 8, 2)
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test1, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 2
iter_num = 2
prev_md = param_iterations[iter_num-1]['max_depth']
prev_mcw = param_iterations[iter_num-1]['min_child_weight']
param_test2 = {
    'max_depth': [prev_md-1, prev_md, prev_md+1],
    'min_child_weight': [
        prev_mcw-1.5, 
        prev_mcw-1, 
        prev_mcw-0.5, 
        prev_mcw, 
        prev_mcw+0.5, 
        prev_mcw+1,
        prev_mcw+1.5,
    ]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test2, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 3
iter_num = 3
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test3, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 4
iter_num = 4
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test4, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 5
iter_num = 5
prev_sub = int(param_iterations[iter_num-1]['subsample'] * 100)
prev_csb = int(param_iterations[iter_num-1]['colsample_bytree'] * 100)
param_test5 = {
    'subsample':[i/100.0 for i in range(prev_sub-10,prev_sub+15,5)],
    'colsample_bytree':[i/100.0 for i in range(prev_csb-10,prev_csb+15,5)]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test5, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 6
iter_num = 6
param_test6 = {
    'reg_alpha':[0, 1e-5, 1e-2, 0.1, 1, 100]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test6, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 7
iter_num = 7
prev_a = param_iterations[iter_num-1]['reg_alpha']
param_test7 = {
    'reg_alpha':[prev_a/2.0, prev_a, prev_a*1.5, prev_a*2.0, prev_a*5.0, prev_a*10.0]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test7, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 8
iter_num = 8
param_test8 = {
    'learning_rate':[0.1, 0.01, 0.001],
    'n_estimators':[100, 200, 300, 400, 500, 600, 700, 800, 900]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test8, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

# %%
# grid search 9
iter_num = 9
prev_ne = param_iterations[iter_num-1]['n_estimators']
param_test9 = {
    'n_estimators':[i for i in range(prev_ne-80, prev_ne+90, 10)]
}
while True:
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test9, group_kfold)
    print('best score is ', scores[iter_num])
    if scores[iter_num] < scores[iter_num - 1]:
        print('score dropped')
        continue
    print('score better')
    break
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# graph feature importance
iter_num = 9
graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# params

params = {}


params[326] = {
    'num_class': 5, 
    'learning_rate': 0.001, 
    'n_estimators': 620, 
    'max_depth': 4, 
    'min_child_weight': 0, 
    'gamma': 0.0, 
    'subsample': 0.8, 
    'colsample_bytree': 0.8, 
    'objective': 'multi:softprob', 
    'nthread': 4, 
    'seed': 326, 
    'verbosity': 0, 
    'use_label_encoder': False, 
    'reg_alpha': 0.0
}