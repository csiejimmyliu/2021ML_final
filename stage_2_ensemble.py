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
# feature selection
xgb_stage2 = XGBClassifier(
    num_class=5,
    learning_rate=0.01,
    n_estimators=410,
    max_depth=5,
    min_child_weight=3,
    gamma=0.3,
    subsample=0.5,
    colsample_bytree=0.5,
    objective='multi:softprob',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False,
    reg_alpha= 0.1
)
xgb_stage2.fit(X_res, y_res)
feat_imp = pd.Series(xgb_stage2.get_booster().get_fscore()).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Feature Importances')
X_res_amputated = X_res[feat_imp.keys()[0:29]].copy()

#%%
# params list
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

param_list = [params1126, params110, params1115, params1124, params326]

#%%
# model list
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

model_list = [
    ('1126', get_model(params1126)),
    ('326', get_model(params326)),
    ('1115', get_model(params1115)),
    ('110', get_model(params110)),
    ('1124', get_model(params1124))
]

#%%
xgb_voting = VotingClassifier(estimators=model_list, voting='hard')
cv_results = cross_validate(xgb_voting, X_res_amputated, y_res, 
    cv=group_kfold.split(X_res, y_res, groups), scoring='f1_micro')
cv_results['test_score']

#%%
# stage 2 predictors
feat_imp.keys()
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
xgb_voting.fit(X_res, y_res)

#%%
xgb_voting.save_model('stage_2_voting.json')

# %%
