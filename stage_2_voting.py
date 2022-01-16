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
SEED = 1124

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
if WITH_GROUPING:
    train_1to5['Group Label'] = np.array(list(range(1108)))
train_1to5['Churn Category'].value_counts()

#%%
# import models
model_list = []

#%%
