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
#final model
xgb_stage2 = XGBClassifier(
    num_class=5,
    learning_rate=0.01,
    n_estimators=410,
    max_depth=6,
    min_child_weight=6.3,
    gamma=0.1,
    subsample=0.5,
    colsample_bytree=0.9,
    objective='multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False,
    reg_alpha= 0.01
)
xgb_stage2.fit(X_res, y_res)
feat_imp = pd.Series(xgb_stage2.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
plt.ylabel('Feature Importance Score')

#%%
score_w_i_features = {}
for i in range(10, 80, 2):
    X_res_amputated = X_res[feat_imp.keys()[0:i]].copy()
    cv_results = cross_validate(xgb_stage2, X_res_amputated, 
        y_res, cv=group_kfold.split(X_res, y_res, groups), scoring='accuracy')
    avg_test_score = np.mean(cv_results['test_score'])
    print(f'performance with first {i} ', avg_test_score)
    score_w_i_features[i] = avg_test_score

# %%
score_w_i_features = pd.Series(score_w_i_features)
score_w_i_features.plot()

# %%
score_w_i_features = {}
for i in range(20, 40):
    X_res_amputated = X_res[feat_imp.keys()[0:i]].copy()
    cv_results = cross_validate(xgb_stage2, X_res_amputated, 
        y_res, cv=group_kfold.split(X_res, y_res, groups), scoring='accuracy')
    avg_test_score = np.mean(cv_results['test_score'])
    print(f'performance with first {i} ', avg_test_score)
    score_w_i_features[i] = avg_test_score
# %%
score_w_i_features = pd.Series(score_w_i_features)
score_w_i_features.plot()

#%%
X_res_amputated = X_res[feat_imp.keys()[0:29]].copy()