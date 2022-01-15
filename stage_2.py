#%%
# Import library
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
# category 1 -> 5
train_1to5 = train.loc[train['Churn Category']  != 0]
train_1to5.loc[train_1to5['Churn Category'] == 5, 'Churn Category'] = 0
train_1to5['Group Label'] = np.array(list(range(1108)))
train_1to5.shape

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
xgb_hyper_params = {
    'num_class': 5,
    'learning_rate': 0.1,
    'n_estimators': 105,
    'max_depth': 5,
    'min_child_weight': 1,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'nthread': 4,
    'seed': 1126,
    'verbosity': 0,
    'use_label_encoder': False
}

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
print('cv performance:', cv_model(xgb_hyper_params, X_res, y_res, group_kfold.split(X_res, y_res, groups)))
graph_imp(xgb_hyper_params, X_res, y_res)

# %%
# grid search 1
param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
}
to_update = grid_search(xgb_hyper_params, X_res, y_res, param_test1, group_kfold.split(X_res, y_res, groups))
xgb_hyper_params2 = xgb_hyper_params.copy()
print(to_update)
for key, value in to_update.items():
    xgb_hyper_params2[key] = value

#%%
cv_model(xgb_hyper_params2, X_res, y_res, group_kfold.split(X_res, y_res, groups))
graph_imp(xgb_hyper_params2, X_res, y_res)

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
#%%
xgb6.fit(X_res, y_res)

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
stage_2_test_data[target] = xgb6.predict(stage_2_test_data[predictors])

#%%
stage_2_test_data[target].replace(0, 5, inplace=True)

#%%
stage_2_test_data[['Customer ID', 'Churn Category']].to_csv('./prediction/stage_2_label.csv', index=False)
