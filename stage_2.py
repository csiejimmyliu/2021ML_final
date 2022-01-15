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

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams

rcParams['figure.figsize'] = 12, 4

APPLY_NORMALIZATION = True

#%%
# import data
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
# drop category 0, category 1 -> 5
train_1to5 = train.loc[train['Churn Category']  != 0].copy()
train_1to5['Churn Category'].replace(5, 0, inplace=True)
train_1to5['Group Label'] = np.array(list(range(1108)))
train_1to5['Churn Category'].value_counts()

# %%
# random oversample and grouping, prevent duplicate examples from appearing in both training and validation sets
oversample = RandomOverSampler()
X = train_1to5[predictors+['Group Label']]
y = train_1to5[target]
X_res, y_res = oversample.fit_resample(X, y)
print(y_res.value_counts())
groups = np.array(X_res['Group Label'])
X_res.drop('Group Label', axis=1, inplace=True)
# X_res.columns
group_kfold = StratifiedGroupKFold(n_splits=5)
# group_kfold.split(X_res, y_res, groups)

#%%
# xgboost hyperparameters
initial_params = {
    'num_class': 5,
    'learning_rate': 0.1,
    'n_estimators': 105,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'nthread': 16,
    'seed': 1126,
    'verbosity': 0,
    'use_label_encoder': False
}
param_iterations = [initial_params] * 10

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
        use_label_encoder=params['use_label_encoder']
    )
    cv_results = cross_validate(alg, data_X, data_y, cv=folds, scoring='accuracy')
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
        use_label_encoder=params['use_label_encoder']
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
        use_label_encoder=original_params['use_label_encoder']
    )
    gsearch = GridSearchCV(estimator=alg, param_grid = param_test, scoring='accuracy',n_jobs=16, cv=folds)
    gsearch.fit(data_X, data_y)
    return gsearch.best_params_


# %%
# initial model performance
iter_num = 0
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
print('initial params: ', param_iterations[iter_num])
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 1
iter_num = 1
param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test1, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 1
iter_num = 1
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 2
iter_num = 2
param_test2 = {
    'max_depth':range(3, 8),
    'min_child_weight':range(3, 8)
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test2, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 2
iter_num = 2
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 3
iter_num = 3
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test3, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 3
iter_num = 3
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)


# %%
# grid search 4
iter_num = 4
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test4, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 4
iter_num = 4
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 5
iter_num = 5
param_test5 = {
    'subsample':[i/100.0 for i in range(55,70,5)],
    'colsample_bytree':[i/100.0 for i in range(85,100,5)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test5, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 5
iter_num = 5
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 6
iter_num = 6
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test6, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 6
iter_num = 6
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 7
iter_num = 7
param_test7 = {
    'reg_alpha':[0.5, 1, 1.5, 2, 5, 10]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test7, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 7
iter_num = 7
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)


# %%
# grid search 8
iter_num = 8
param_test8 = {
    'learning_rate':[0.1, 0.01, 0.001],
    'n_estimators':[100, 200, 300, 400, 500, 600, 700]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test8, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()


#%%
# performance after grid search 8
iter_num = 8
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 9
iter_num = 9
param_test9 = {
    'n_estimators':[i for i in range(120, 290, 10)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, param_test9, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()


#%%
# performance after grid search 9
iter_num = 9
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, group_kfold.split(X_res, y_res, groups)))
graph_imp(param_iterations[iter_num], X_res, y_res)

#%%
#final model
xgb_stage2 = XGBClassifier(
    num_class=5,
    learning_rate=0.001,
    n_estimators=150,
    max_depth=6,
    min_child_weight=3,
    gamma=0.0,
    subsample=0.65,
    colsample_bytree=0.9,
    objective='multi:softprob',
    nthread=16,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)

#%%
stage_1_result = pd.read_csv('./prediction/stage_1_label.csv')
stage_2_id = stage_1_result.loc[stage_1_result['Churn Category'] == 1][['Customer ID']]

#%%
stage_2_test_data = pd.merge(
    left=stage_2_id,
    right=test_data,
    how='left',
    on='Customer ID'
)
stage_2_test_data

#%%
stage_2_test_data[target] = xgb_stage2.predict(stage_2_test_data[predictors])

#%%
stage_2_test_data[target].replace(0, 5, inplace=True)

#%%
stage_2_test_data[['Customer ID', 'Churn Category']].to_csv('./prediction/stage_2_label.csv', index=False)
