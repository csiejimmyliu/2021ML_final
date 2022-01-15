#%%
# import library
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from imblearn.over_sampling import RandomOverSampler

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
from sklearn.ensemble import VotingClassifier

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
# feature list
stage_2_predictors = ['Longitude', 'Latitude', 'Population',
    'Avg Monthly Long Distance Charges', 'Monthly Charge', 'Age',
    'Total Charges', 'Avg Monthly GB Download',
    'Total Long Distance Charges', 'Total Revenue', 'Tenure in Months',
    'Satisfaction Score', 'Gender_Male', 'Total Extra Data Charges',
    'Gender_Female', 'Number of Referrals', 'Married_No', 'Offer_None',
    'Paperless Billing_Yes', 'Premium Tech Support_No',
    'Online Security_No', 'Paperless Billing_No', 'Married_nan',
    'Multiple Lines_Yes', 'Online Backup_No', 'Streaming Music_No',
    'Device Protection Plan_No', 'Married_Yes', 'Streaming TV_Yes',
    'Total Refunds', 'Offer_Offer E', 'Contract_Month-to-Month',
    'Internet Type_Fiber Optic', 'Device Protection Plan_Yes',
    'Streaming Movies_No', 'Streaming TV_No', 'Multiple Lines_No',
    'Gender_nan', 'Streaming Music_Yes', 'Online Security_Yes',
    'Payment Method_Bank Withdrawal', 'Unlimited Data_No',
    'Payment Method_Credit Card', 'Premium Tech Support_nan',
    'Streaming TV_nan', 'Device Protection Plan_nan', 'Internet Type_Cable',
    'Phone Service_Yes', 'Streaming Movies_Yes', 'Online Backup_Yes',
    'Online Backup_nan', 'Internet Service_Yes', 'Streaming Music_nan',
    'Payment Method_nan', 'Internet Service_No', 'Contract_nan',
    'Internet Type_DSL', 'Paperless Billing_nan', 'Offer_Offer D',
    'Multiple Lines_nan', 'Offer_nan', 'Streaming Movies_nan',
    'Online Security_nan', 'Internet Type_nan', 'Premium Tech Support_Yes',
    'Payment Method_Mailed Check', 'Contract_One Year',
    'Unlimited Data_nan', 'Phone Service_nan', 'Number of Dependents',
    'Offer_Offer C', 'Internet Service_nan', 'Unlimited Data_Yes',
    'Internet Type_None', 'Phone Service_No', 'Offer_Offer B',
    'Contract_Two Year']

#%%
# random forest
rf = RandomForestClassifier()
cv_results = cross_validate(rf, X_res, y_res, 
    cv=group_kfold.split(X_res, y_res, groups), scoring='f1_micro')
np.average(cv_results['test_score'])

# %%
# adaboost
adb = AdaBoostClassifier()
cv_results = cross_validate(adb, X_res, y_res, 
    cv=group_kfold.split(X_res, y_res, groups), scoring='f1_micro')
np.average(cv_results['test_score'])
# %%
