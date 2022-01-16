#%%
# import library
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from imblearn.over_sampling import RandomOverSampler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 12, 4

WITH_GROUPING = True
SEED_LIST = [1126, 326, 1115, 110, 1124, 6, 7, 8, 9]

#%%
# function definition
def grid_search(original_params, data_X, data_y, param_test, folds, g):
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
        scoring='f1_micro',n_jobs=16, cv=folds.split(data_X, data_y, g))
    gsearch.fit(data_X, data_y)
    return gsearch.best_score_, gsearch.best_params_

def get_model(params):
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
    return alg

#%%
# import data
train = pd.read_csv('./preprocessed_data/train_data_std_normalized.csv')
train = train.loc[train['Churn Category'] != -1]
test_data = pd.read_csv('./preprocessed_data/test_data_std_normalized.csv')
target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]
NUM_CLASS = len(train[target].unique())

#%%
# drop category 0, category No. minus 1
train_1to5 = train.loc[train[target]  != 0].copy()
for i in range(1,6):
    train_1to5[target].replace(i, i-1, inplace=True)
train_1to5[target].value_counts()
X = train_1to5[predictors]
y = train_1to5[target]

for ITERATION in range(7):
    SEED = SEED_LIST[ITERATION]
    print("######################################")
    print("# Iteration: ", ITERATION)
    print("######################################")

    # # sample 1/3 of each category
    # kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    # train, val = next(kfold.split(X, y))
    # X_sample = X.iloc[val, :]
    # y_sample = y.iloc[val]
    # y_sample.value_counts()

    # random oversample and grouping
    # prevent duplicate examples from appearing in both training and validation sets
    X['Group Label'] = np.array(list(range(len(X))))
    oversample = RandomOverSampler(random_state=SEED)
    X_res, y_res = oversample.fit_resample(X, y)
    print(y_res.value_counts())
    groups = np.array(X_res['Group Label'])
    X_res.drop('Group Label', axis=1, inplace=True)
    group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)


    # initial xgboost hyperparameters
    TUNING_STAGES = 7
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

    # find initial n_estimators
    xgtrain = xgb.DMatrix(X_res.values, y_res.values)
    cvresult = xgb.cv(initial_params, xgtrain, num_boost_round=1000, 
        nfold=5, stratified=True,
        metrics='merror', early_stopping_rounds=50, verbose_eval=False)
    print('initial n_estimaotrs: ', cvresult.shape[0])
    param_iterations[0]['n_estimators'] = cvresult.shape[0]

    # grid search 1
    print("grid search 1")
    iter_num = 1
    param_test1 = {
        'max_depth':range(3, 10, 3),
        'min_child_weight':range(1, 8, 3)
    }

    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test1, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # grid search 2
    print("grid search 2")
    iter_num = 2
    prev_md = param_iterations[iter_num-1]['max_depth']
    prev_mcw = param_iterations[iter_num-1]['min_child_weight']
    param_test2 = {
        'max_depth': [prev_md-1, prev_md, prev_md+1],
        'min_child_weight': [
            prev_mcw-1, 
            prev_mcw-0.5, 
            prev_mcw, 
            prev_mcw+0.5, 
            prev_mcw+1,
        ]
    }

    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test2, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # grid search 3
    print("grid search 3")
    iter_num = 3
    param_test3 = {
        'gamma':[i/10.0 for i in range(0,4)],
    }
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test3, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # grid search 4
    print("grid search 4")
    iter_num = 4
    param_test4 = {
        'subsample':[i/10.0 for i in range(6,11,2)],
        'colsample_bytree':[i/10.0 for i in range(6,11,2)]
    }
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test4, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # grid search 5
    print("grid search 5")
    iter_num = 5
    prev_sub = int(param_iterations[iter_num-1]['subsample'] * 100)
    prev_csb = int(param_iterations[iter_num-1]['colsample_bytree'] * 100)
    param_test5 = {
        'subsample':[i/100.0 for i in range(prev_sub-10,prev_sub+15,5)],
        'colsample_bytree':[i/100.0 for i in range(prev_csb-10,prev_csb+15,5)]
    }
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test5, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # grid search 6
    print("grid search 6")
    iter_num = 6
    param_test6 = {
        'learning_rate':[0.1, 0.01, 0.001],
        'n_estimators':[200, 400, 600]
    }
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test6, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # grid search 7
    print("grid search 7")
    iter_num = 7
    prev_lr = param_iterations[iter_num-1]['learning_rate']
    prev_ne = param_iterations[iter_num-1]['n_estimators']
    param_test7 = {
        'learning_rate':[prev_lr/2, prev_lr, prev_lr*2],
        'n_estimators':[i for i in range(prev_ne-100, prev_ne+110, 50)]
    }
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
        param_test7, group_kfold, groups)
    # print('best score is ', scores[iter_num])
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    # print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()

    # model fit
    stage_2_model = get_model(param_iterations[TUNING_STAGES])
    stage_2_model.fit(X_res, y_res, eval_metric='merror')

    # save_model
    stage_2_model.save_model(f'./stage_2_final_ensemble/stage_2_bagging_model_{ITERATION}.json')

