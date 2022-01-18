#%%
# import library
import numpy as np
import pandas as pd

from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

from imblearn.over_sampling import RandomOverSampler

#%%
# import data
train = pd.read_csv('./preprocessed_data/train_data_std_normalized.csv')
train = train.loc[train['Churn Category'] != -1]
test_data = pd.read_csv('./preprocessed_data/test_data_std_normalized.csv')
TARGET = 'Churn Category'
IDCOL = 'Customer ID'
PREDICTORS = [x for x in train.columns if x not in [TARGET, IDCOL]]
X = train[PREDICTORS]
y = train[TARGET]

#%%
# stage 1 oversampling
temp_train=train.copy()

train_0=temp_train.loc[temp_train['Churn Category'] == 0]
train_0=train_0.sample(n=2500,replace=False)

num_list=[]
for i in range(6):
    num_list.append(sum(temp_train['Churn Category']==i))

N=len(train_0)/5
train_stage1=train_0

for i in range(1,6):
    train_stage1=train_stage1.append(temp_train.loc[temp_train['Churn Category'] == i])

train_stage1['Group Label'] = np.array(list(range(len(train_stage1))))

for i in range(1,6):
    train=train.append(temp_train.loc[temp_train['Churn Category'] == i].sample(n=int(N-num_list[i]),replace=True))

groups1 = np.array(train_stage1['Group Label'])
train_stage1.drop('Group Label', axis=1, inplace=True)
train_stage1['Churn Category']=np.where(train_stage1[TARGET]==0,0,1)
X1 = train_stage1[PREDICTORS]
y1 = train_stage1[TARGET]

#%%
# stage 2 oversampling
train_stage2 = train.loc[train[TARGET]!=0]
for i in range(1,6):
    train_stage2.loc[train_stage2[TARGET]==i] = i-1

train_stage2['Group Label'] = np.array(list(range(len(train_stage2))))
oversample = RandomOverSampler()
X2 = train_stage2[PREDICTORS+['Group Label']]
y2 = train_stage2[TARGET]
X2, y2 = oversample.fit_resample(X2, y2)
print(y2.value_counts())

groups2 = np.array(X2['Group Label'])
X2.drop('Group Label', axis=1, inplace=True)
group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)


# %%
# version
print(
    """
    ###############################################
    #     all in one SVM w REF kernel w weight    #
    ###############################################
    """
)
svm = SVC(C=1, gamma=0.1, kernel='rbf', class_weight='balanced')

#%%
# cross valudation with SVM
kfold = StratifiedKFold(n_splits=5, shuffle=False)
cv_results = cross_validate(svm, X, y, cv=kfold.split(X,y), scoring='f1_macro')
print("RBFSVM cv f1-macro score:", np.average(cv_results['test_score']))

#%%
# grid search
param_test_svm = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1]
}
gsearch = GridSearchCV(estimator=svm, param_grid=param_test_svm, 
        scoring='f1_macro',n_jobs=16, cv=kfold.split(X,y))
gsearch.fit(X, y)
print(gsearch.best_score_, gsearch.best_params_)
svm = SVC(C=gsearch.best_params_['C'], gamma=gsearch.best_params_['gamma'], kernel='rbf', class_weight='balanced')

#%%
# grid search again
param_test_svm = {
    'C': [0.2, 0.5, 1, 2, 5],
    'gamma': [0.02, 0.05, 0.1, 0.2, 0.5]
}
gsearch = GridSearchCV(estimator=svm, param_grid=param_test_svm, 
        scoring='f1_macro',n_jobs=16, cv=kfold.split(X,y))
gsearch.fit(X, y)
print(gsearch.best_score_, gsearch.best_params_)
svm = SVC(C=gsearch.best_params_['C'], gamma=gsearch.best_params_['gamma'], kernel='rbf', class_weight='balanced')

#%%
# grid search again again
param_test_svm = {
    'C': [0.2, 0.5, 1, 2, 5, 8, 10, 15],
    'gamma': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
}
gsearch = GridSearchCV(estimator=svm, param_grid=param_test_svm, 
        scoring='f1_macro',n_jobs=16, cv=kfold.split(X,y))
gsearch.fit(X, y)
print(gsearch.best_score_, gsearch.best_params_)
svm = SVC(C=gsearch.best_params_['C'], gamma=gsearch.best_params_['gamma'], kernel='rbf', class_weight='balanced')

#%%
# train and predict
X = train[PREDICTORS]
y = train[TARGET]
svm_allinone = svm.fit(X, y)
y_test = svm_allinone.predict(test_data[PREDICTORS])

# %%
# output csv
test_result = test_data
test_result[TARGET] = y_test
test_result[[IDCOL, TARGET]].to_csv('./toy_outputs/svm_allinone_weight.csv', index=False)

# %%
# version
print(
    """
    ###############################################
    #      svm rbf two stages w oversampling      #
    ###############################################
    """
)

#%%
# stage 1 grid seacrh 1
svm1 = SVC(C=1, gamma=0.1, kernel='rbf')
# param_test_svm1_1 = {
#     'C': [0.01, 0.1, 1, 10],
#     'gamma': [0.001, 0.01, 0.1, 1]
# }
# gsearch1_1 = GridSearchCV(estimator=svm1, param_grid=param_test_svm1_1, 
#         scoring='f1_macro',n_jobs=16, cv=group_kfold.split(X1,y1,groups1))
# gsearch1_1.fit(X1, y1)
# print(gsearch1_1.best_score_, gsearch1_1.best_params_)
# svm1 = SVC(C=gsearch1_1.best_params_['C'], gamma=gsearch1_1.best_params_['gamma'], kernel='rbf')

# #%%
# # stage 1 grid seacrh 2
# svm1 = SVC(C=1, gamma=0.1, kernel='rbf')
# param_test_svm1_2 = {
#     'C': [2, 5, 10, 20, 50],
#     'gamma': [0.002, 0.005, 0.01, 0.02, 0.05]
# }
# gsearch1_2 = GridSearchCV(estimator=svm1, param_grid=param_test_svm1_2, 
#         scoring='f1_macro',n_jobs=16, cv=group_kfold.split(X1,y1,groups1))
# gsearch1_2.fit(X1, y1)
# print(gsearch1_2.best_score_, gsearch1_2.best_params_)
# svm1 = SVC(C=gsearch1_2.best_params_['C'], gamma=gsearch1_2.best_params_['gamma'], kernel='rbf')

#%%
# stage 2 grid seacrh 1
svm2 = SVC(C=1, gamma=0.1, kernel='rbf')
param_test_svm2_1 = {
    'C': [0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1]
}
gsearch2_1 = GridSearchCV(estimator=svm2, param_grid=param_test_svm2_1, 
        scoring='f1_macro',n_jobs=16, cv=group_kfold.split(X2,y2,groups2))
gsearch2_1.fit(X2, y2)
print(gsearch2_1.best_score_, gsearch2_1.best_params_)
svm2 = SVC(C=gsearch2_1.best_params_['C'], gamma=gsearch2_1.best_params_['gamma'], kernel='rbf')

#%%
# stage 2 grid seacrh 2
svm2 = SVC(C=1, gamma=0.1, kernel='rbf')
param_test_svm2_2 = {
    'C': [0.002, 0.005, 0.01, 0.02, 0.05],
    'gamma': [0.0002, 0.0005, 0.001, 0.002, 0.005]
}
gsearch2_2 = GridSearchCV(estimator=svm2, param_grid=param_test_svm2_2, 
        scoring='f1_macro',n_jobs=16, cv=group_kfold.split(X2,y2,groups2))
gsearch2_2.fit(X2, y2)
print(gsearch2_2.best_score_, gsearch2_2.best_params_)
svm2 = SVC(C=gsearch2_2.best_params_['C'], gamma=gsearch2_2.best_params_['gamma'], kernel='rbf')

# %%
# stage 1 train and predict
svm_stage1 = svm1.fit(X1, y1)
y1_test = svm_stage1.predict(test_data[PREDICTORS])


#%%
# get stage 2 test data
test_result1 = test_data.copy()
test_result1[TARGET] = y1_test
test_stage2 = test_result1.loc[test_result1[TARGET] == 1]

# %%
# stage 2 train and predict
svm_stage2 = svm2.fit(X2, y2)
y2_test = svm_stage2.predict(test_stage2[PREDICTORS])
test_result2 = test_stage2.copy()
test_result2[TARGET] = y2_test + 1

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
test_result[[IDCOL, TARGET]].to_csv('./toy_outputs/svm_2stages_os.csv', index=False)
# %%