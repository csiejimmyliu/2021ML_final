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
WITH_GROUPING = True
SEED = 1126

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
# drop category 0, category No. minus 1
train_1to5 = train.loc[train['Churn Category']  != 0].copy()
for i in range(1,6):
    train_1to5['Churn Category'].replace(i, i-1, inplace=True)
if WITH_GROUPING:
    train_1to5['Group Label'] = np.array(list(range(1108)))
train_1to5['Churn Category'].value_counts()

# %%
# random oversample and grouping, prevent duplicate examples from appearing in both training and validation sets
oversample = RandomOverSampler(random_state=SEED)
if WITH_GROUPING:
    X = train_1to5[predictors+['Group Label']]
else:
    X = train_1to5[predictors]
y = train_1to5[target]
X_res, y_res = oversample.fit_resample(X, y)
print(y_res.value_counts())
if WITH_GROUPING:
    groups = np.array(X_res['Group Label'])
    X_res.drop('Group Label', axis=1, inplace=True)
    group_kfold = StratifiedGroupKFold(n_splits=5)

#%%
# xgboost hyperparameters
TUNING_STAGES = 11
initial_params = {
    'num_class': 5,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
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
    cv_results = cross_validate(alg, data_X, data_y, cv=folds, scoring='f1_micro')
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
    gsearch = GridSearchCV(estimator=alg, param_grid = param_test, scoring='f1_micro',n_jobs=16, cv=folds)
    gsearch.fit(data_X, data_y)
    return gsearch.best_params_


# %%
# initial model performance
iter_num = 0
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
print('initial params: ', param_iterations[iter_num])
# graph_imp(param_iterations[iter_num], X_res, y_res)

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
    'max_depth':range(3, 10),
    'min_child_weight':range(1, 7)
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test1, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 1
iter_num = 1
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 2
iter_num = 2
prev_mcw = param_iterations[iter_num-1]['min_child_weight']
param_test2 = {
    'min_child_weight': [prev_mcw-0.5, prev_mcw-0.3, prev_mcw, prev_mcw+0.3, prev_mcw+0.5]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test2, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 2
iter_num = 2
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 3
iter_num = 3
param_test3 = {
    'gamma':[i/10.0 for i in range(0,5)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test3, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 3
iter_num = 3
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 4
iter_num = 4
param_test4 = {
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test4, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 4
iter_num = 4
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 5
iter_num = 5
prev_sub = int(param_iterations[iter_num-1]['subsample'] * 100)
prev_csb = int(param_iterations[iter_num-1]['colsample_bytree'] * 100)
param_test5 = {
    'subsample':[i/100.0 for i in range(prev_sub-10,prev_sub+15,5)],
    'colsample_bytree':[i/100.0 for i in range(prev_csb-10,prev_csb+15,5)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test5, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 5
iter_num = 5
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 6
iter_num = 6
param_test6 = {
    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test6, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 6
iter_num = 6
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 7
iter_num = 7
prev_a = param_iterations[iter_num-1]['reg_alpha']
param_test7 = {
    'reg_alpha':[prev_a/2.0, prev_a, prev_a*1.5, prev_a*2.0, prev_a*5.0, prev_a*10.0]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test7, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()

#%%
# performance after grid search 7
iter_num = 7
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)


# %%
# grid search 8
iter_num = 8
param_test8 = {
    'learning_rate':[0.1, 0.01, 0.001],
    'n_estimators':[100, 200, 300, 400, 500, 600, 700]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test8, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()


#%%
# performance after grid search 8
iter_num = 8
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
# graph_imp(param_iterations[iter_num], X_res, y_res)

# %%
# grid search 9
iter_num = 9
prev_ne = param_iterations[iter_num-1]['n_estimators']
param_test9 = {
    'n_estimators':[i for i in range(prev_ne-80, prev_ne+90, 10)]
}
to_update = grid_search(param_iterations[iter_num], X_res, y_res, 
    param_test9, group_kfold.split(X_res, y_res, groups))
new_params = param_iterations[iter_num-1].copy()
for key, value in to_update.items():
    new_params[key] = value
print('best params: ', to_update, '\n' ,new_params)
param_iterations[iter_num] = new_params.copy()


#%%
# performance after grid search 9
iter_num = 9
print(param_iterations[iter_num])
print('cv performance:', cv_model(param_iterations[iter_num], X_res, y_res, 
    group_kfold.split(X_res, y_res, groups)))
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
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False,
    reg_alpha = 2
)
xgb_stage2.fit(X_res, y_res, eval_metric='merror')
feat_imp = pd.Series(xgb_stage2.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

params1126 = {
    'num_class': 5, 
    'learning_rate': 0.01, 
    'n_estimators': 410, 
    'max_depth': 5, 
    'min_child_weight': 3, 
    'gamma': 0.3, 
    'subsample': 0.5, 
    'colsample_bytree': 0.5, 
    'objective': 'multi:softprob', 
    'nthread': 4, 
    'seed': 1126, 
    'verbosity': 0, 
    'use_label_encoder': False, 
    'reg_alpha': 0.5
}

params326 =  {
    'num_class': 5, 
    'learning_rate': 0.001, 
    'n_estimators': 640, 
    'max_depth': 8, 
    'min_child_weight': 6, 
    'gamma': 0.3, 
    'subsample': 0.7, 
    'colsample_bytree': 0.85, 
    'objective': 'multi:softprob', 
    'nthread': 4, 
    'seed': 326, 
    'verbosity': 0,
    'use_label_encoder': False, 
    'reg_alpha': 0.1
}

params1115 = {
    'num_class': 5, 
    'learning_rate': 0.01, 
    'n_estimators': 460, 'max_depth': 3, 
    'min_child_weight': 3, 
    'gamma': 0.4, 
    'subsample': 0.6, 
    'colsample_bytree': 0.6, 
    'objective': 'multi:softprob', 
    'nthread': 4, 
    'seed': 1115, 
    'verbosity': 0, 
    'use_label_encoder': False, 
    'reg_alpha': 10.0
}

params110 = {
    'num_class': 5, 
    'learning_rate': 0.01, 
    'n_estimators': 560, 
    'max_depth': 6, 
    'min_child_weight': 7, 
    'gamma': 0.0, 
    'subsample': 0.9, 
    'colsample_bytree': 0.75, 
    'objective': 'multi:softprob', 
    'nthread': 4, 
    'seed': 110, 
    'verbosity': 0, 
    'use_label_encoder': False, 
    'reg_alpha': 5e-06
}

params1124 = {
    'num_class': 5, 
    'learning_rate': 0.001, 
    'n_estimators': 620, 
    'max_depth': 7, 
    'min_child_weight': 5, 
    'gamma': 0.4, 
    'subsample': 0.65, 
    'colsample_bytree': 0.9, 
    'objective': 'multi:softprob', 
    'nthread': 4, 
    'seed': 1124, 
    'verbosity': 0, 
    'use_label_encoder': False, 
    'reg_alpha': 0.01
}

#%%
# save_model
xgb_stage2.save_model('stage_2_v1.json')

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
