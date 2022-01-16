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
SEED_LIST = list(range(0, 63, 3))

#%%
# import data
train = pd.read_csv('./preprocessed_data/train_data_std_normalized.csv')
train = train.loc[train['Churn Category'] != -1]
test_data = pd.read_csv('./preprocessed_data/test_data_std_normalized.csv')
target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]

#%%
# drop category 0, category No. minus 1
train_1to5 = train.loc[train[target]  != 0].copy()
for i in range(1,6):
    train_1to5[target].replace(i, i-1, inplace=True)
train_1to5[target].value_counts()
X = train_1to5[predictors]
y = train_1to5[target]

for ITERATION in range(21):
    SEED = SEED_LIST[ITERATION]
    print("######################################")
    print("# Iteration: ", ITERATION)
    print("######################################")
    
    X['Group Label'] = np.array(list(range(len(X))))
    oversample = RandomOverSampler(random_state=SEED)
    X_res, y_res = oversample.fit_resample(X, y)
    print(y_res.value_counts())
    groups = np.array(X_res['Group Label'])
    X_res.drop('Group Label', axis=1, inplace=True)
    group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)

    alg = XGBClassifier(
        num_class= 5, 
        learning_rate= 0.001, 
        n_estimators= 170, 
        max_depth= 3, 
        in_child_weight= 4, 
        gamma= 0.1, 
        subsample= 0.55, 
        colsample_bytree= 0.6, 
        objective= 'multi:softprob', 
        nthread= 4, 
        seed= SEED, 
        verbosity= 0, 
        use_label_encoder= False, 
        reg_alpha= 0
    )

    alg.fit(X_res, y_res, eval_metric='merror')

    # save_model
    alg.save_model(f'./stage_2_final_final_ensemble/stage_2_final_model_{ITERATION}.json')
