#%%
# import libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedGroupKFold

import joblib

#%%
# import data
train = pd.read_csv('./preprocessed_data/train_data_std_normalized.csv')
train = train.loc[train['Churn Category'] != -1]
test_data = pd.read_csv('./preprocessed_data/test_data_std_normalized.csv')

TARGET = 'Churn Category'
IDCOL = 'Customer ID'
PREDICTORS = [x for x in train.columns if x not in [TARGET, IDCOL]]

#%%
# stage 1 oversampling
temp_train=train.copy()
train_0=temp_train.loc[temp_train['Churn Category'] == 0]

num_list=[]
for i in range(6):
    num_list.append(sum(temp_train['Churn Category']==i))

N=len(train_0)/5
train_stage1=train_0

for i in range(1,6):
    train_stage1=train_stage1.append(temp_train.loc[temp_train['Churn Category'] == i])

train_stage1['Group Label'] = np.array(list(range(len(train_stage1))))

for i in range(1,6):
    train_stage1=train_stage1.append(temp_train.loc[temp_train['Churn Category'] == i].sample(n=int(N-num_list[i]),replace=True))

groups1 = np.array(train_stage1['Group Label'])
train_stage1.drop('Group Label', axis=1, inplace=True)
train_stage1['Churn Category']=np.where(train_stage1[TARGET]==0,0,1)
group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)
X1 = train_stage1[PREDICTORS]
y1 = train_stage1[TARGET]

#%%
# stage 1 tuning 1
# rmc1= RandomForestClassifier(n_estimators=200)
# param_test1_1 = {
#     'bootstrap': [True, False],
#     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'max_features': ['auto', 'sqrt'],
#     'min_samples_leaf': [1, 2, 4],
#     'min_samples_split': [2, 5, 10],
#     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# }
# grandom1_1 = RandomizedSearchCV(estimator=rmc1, param_distributions=param_test1_1, n_iter=1000, scoring='f1',
#     cv=group_kfold.split(X1,y1,groups1), verbose=2, n_jobs=16)
# grandom1_1.fit(X1, y1)
# print(grandom1_1.best_score_, grandom1_1.best_params_)
rmc1= RandomForestClassifier(
    n_estimators=200,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=False,
    max_depth=100
)

#%%
# stage 1 tuning 2
# param_test1_2 = {
#     'n_estimators': [100, 200, 300],
#     'min_samples_split': [2, 3],
#     'min_samples_leaf': [1, 2, 3],
#     'max_depth': [80, 100, 120],
# }
# gsearch1_2 = GridSearchCV(estimator=rmc1, param_grid=param_test1_2, scoring='f1',
#     cv=group_kfold.split(X1,y1,groups1), verbose=2, n_jobs=16)
# gsearch1_2.fit(X1, y1)
# print(gsearch1_2.best_score_, gsearch1_2.best_params_)


#%%
# stage 2 oversampling
train_stage2 = train.loc[train[TARGET]!=0].copy()
for i in range(1,6):
   train_stage2[TARGET].replace(i, i-1, inplace=True)

train_stage2['Group Label'] = np.array(list(range(len(train_stage2))))
oversample = RandomOverSampler()
# XX = train_stage2[PREDICTORS+['Group Label']]
XX = train_stage2[PREDICTORS]
yy = train_stage2[TARGET]
X2, y2 = oversample.fit_resample(XX, yy)
print(y2.value_counts())

# groups2 = np.array(X2['Group Label'])
# X2.drop('Group Label', axis=1, inplace=True)

#%%
# stage 2 tuning 1
# rmc2=RandomForestClassifier(n_estimators=300)
# param_test2_1 = {
#     'bootstrap': [True, False],
#     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
#     'max_features': ['auto', 'sqrt'],
#     'min_samples_leaf': [1, 2, 4],
#     'min_samples_split': [2, 5, 10],
#     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
# }
# grandom2_1 = RandomizedSearchCV(estimator=rmc2, param_distributions=param_test2_1, n_iter=1000, scoring='f1_macro',
#     cv=group_kfold.split(X2,y2,groups2), verbose=2, n_jobs=16)
# grandom2_1.fit(X2, y2)
# print(grandom2_1.best_score_, grandom2_1.best_params_)
rmc2= RandomForestClassifier(
    n_estimators=600,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    max_depth=20,
    bootstrap=False
)


#%%
# stage 2 tuning 2
# param_test2_2 = {
#     'n_estimators': [500, 600, 700],
#     'min_samples_split': [4, 5, 6],
#     'min_samples_leaf': [1, 2, 3],
#     'max_depth': [15, 20, 25],
# }
# gsearch2_2 = GridSearchCV(estimator=rmc2, param_grid=param_test2_2, scoring='f1_macro',
#     cv=group_kfold.split(X2,y2,groups2), verbose=2, n_jobs=16)
# gsearch2_2.fit(X2, y2)
# print(gsearch2_2.best_score_, gsearch2_2.best_params_)

#%% 
# stage 1 final model
rmc1f= RandomForestClassifier(
    n_estimators=200,
    min_samples_split=3,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=False,
    max_depth=80
)
rmc1f= RandomForestClassifier(
    n_estimators=200,
)

#%% 
# stage 1 training
rmc1f=rmc1f.fit(X1,y1)

#%%
# stage 2 final model
rmc2f= RandomForestClassifier(
    n_estimators=500,
    min_samples_split=4,
    min_samples_leaf=1,
    max_features='sqrt',
    max_depth=15,
    bootstrap=False
)

rmc2f= RandomForestClassifier(
    n_estimators=300,
)

#%%
# stage 2 training
rmc2f = rmc2f.fit(X2,y2)
test=rmc2f.predict(X2)
for i in test :
    print(i)
#%%
# stage 1 predict and get 2 test data
y1_test = rmc1f.predict(test_data[PREDICTORS])
test_result1 = test_data.copy()
test_result1[TARGET] = y1_test
test_stage2 = test_result1.loc[test_result1[TARGET] == 1]

#%%
test_stage2[PREDICTORS]

# %%
# stage 2 predict
y2_test = rmc2f.predict(test_stage2[PREDICTORS])
test_result2 = test_stage2.copy()
test_result2[TARGET] = y2_test + 1

#%%
# correctness check
np.unique(y2_test)

#%%
# merge
test_result = pd.merge(
    left=test_result1,
    right=test_result2[[IDCOL,TARGET]],
    how='left',
    on=IDCOL,
    suffixes=('', '_2')
)

for i in range(len(test_result)):
    if test_result['Churn Category'].iloc[i] == 1:
        test_result['Churn Category'].iloc[i] = test_result['Churn Category_2'].iloc[i]

test_result['Churn Category'] = test_result['Churn Category'].astype(int)
test_result[[IDCOL, TARGET]].to_csv('./toy_outputs/rf_w_tuning.csv', index=False)
# %%