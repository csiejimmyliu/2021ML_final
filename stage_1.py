#%%
# Import librarie
import time
import pandas as pd
import numpy as np

from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import StratifiedGroupKFold

from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
from sklearn.metrics import precision_score, make_scorer

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

#stage 1 is binary classification
NUM_CLASS=2

#%%
print(train.shape)

#%%
train['Group Label'] = np.array(list(range(len(train))))
#data augmentation



# X_res.columns
group_kfold = StratifiedGroupKFold(n_splits=5)

num_list=[]
for i in range(6):
    num_list.append(sum(train['Churn Category']==i))

N=num_list[0]/5
for i in range(1,6):
    train=train.append(train.loc[train['Churn Category'] == i].sample(n=int(N-num_list[i]),replace=True))



groups = np.array(train['Group Label'])
train.drop('Group Label', axis=1, inplace=True)
train['Churn Category']=np.where(train[target]==0,0,1)




#%%
print(train['Churn Category'].value_counts())



#%%
def modelfit(alg, dtrain, predictors,useTrainCV=False, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='error', early_stopping_rounds=early_stopping_rounds, verbose_eval=False)
        print(cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain[target],eval_metric='error')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions))
    # print("AUC Score (Train): %f" % metrics.accuracy_score(dtrain[target], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
    
    
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

#%%
#Choose all predictors except target & IDcols
xgb1 = XGBClassifier(
    #num_class=NUM_CLASS,
    #scale_pos_weight=2.81407942238,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=13,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    seed=1126,
    verbosity=0,
    use_label_encoder=False
)
modelfit(xgb1, train, predictors)

#%%
sort_index=pd.Series(xgb1.get_booster().get_fscore()).sort_values(ascending=False).index

gkfold=group_kfold.split(train[predictors],train[target],groups)

#step2

# %%
param_test1 = {
    'max_depth':range(3,15,2),
    'min_child_weight':range(1,6,2)
}
xgb2 = XGBClassifier(
    learning_rate=0.1,
    n_estimators=105,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    #reg_alpha=5.6,
    objective='binary:logistic',
    nthread=4,
    #scale_pos_weight=1,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)

#%%

#predictors=sort_index[:75]
gsearch1 = GridSearchCV(estimator=xgb2,param_grid=param_test1,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch1.fit(train[predictors],train[target])
print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)
# opt = (3, 1)

#%%


# %%
# [-1,+1] of the param_test1 result
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight':[0,1,2]
}
gsearch2 = GridSearchCV(estimator=xgb2,param_grid=param_test2,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch2.fit(train[predictors],train[target])
gsearch2.cv_results_, gsearch2.best_params_, gsearch2.best_score_
# opt = (3, 1)

# %%
# try the bigger 4 number, if the param_test2 value is the max
param_test2b = {
 'min_child_weight':[0.25,0.5,0.75,1,1.25,1.5,1.75]
}
gsearch2b = GridSearchCV(estimator=xgb2,param_grid=param_test2b,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch2b.fit(train[predictors],train[target])
gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_
# opt = 1

# %%
modelfit(gsearch2b.best_estimator_, train, predictors)
gsearch2b.cv_results_, gsearch2b.best_params_, gsearch2b.best_score_

# step 3
#%%
param_test3 = {
 'gamma':[i/10.0 for i in range(0,10)]
}
gsearch3 = GridSearchCV(estimator=xgb2,param_grid=param_test3,scoring='f1',n_jobs=4, cv=group_kfold.split(train[predictors],train[target],groups))
gsearch3.fit(train[predictors],train[target])
gsearch3.cv_results_, gsearch3.best_params_, gsearch3.best_score_
# opt = 0.8

#%%
xgb3 = XGBClassifier(
    #num_class=NUM_CLASS,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=3,
    min_child_weight=1,
    gamma=0.8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)
modelfit(xgb3, train, predictors)

#step 4

# %%
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}
gsearch4 = GridSearchCV(estimator=xgb3,param_grid=param_test4,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch4.fit(train[predictors],train[target])
gsearch4.cv_results_, gsearch4.best_params_, gsearch4.best_score_
# opt = (0.7, 0.7)

#%%
param_test5 = {
 'subsample':[i/100.0 for i in range(55,80,5)],
 'colsample_bytree':[i/100.0 for i in range(55,80,5)]
}
gsearch5 = GridSearchCV(estimator=xgb3,param_grid=param_test5,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch5.fit(train[predictors],train[target])
gsearch5.cv_results_, gsearch5.best_params_, gsearch5.best_score_
# opt = (0.75, 0.7)

#%%
xgb4 = XGBClassifier(
    #num_class=NUM_CLASS,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=3,
    min_child_weight=1,
    gamma=0.8,
    colsample_bytree=0.75,
    subsample=0.7,
    
    objective= 'binary:logistic',
    nthread=4,
    scale_pos_weight=1,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)
modelfit(xgb4, train, predictors)

#%%
param_test6 = {
 'reg_alpha':[1e-5, 1e-2,0, 0.1, 1, 100]
}
gsearch6 = GridSearchCV(estimator=xgb4,param_grid=param_test6,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch6.fit(train[predictors],train[target])
gsearch6.cv_results_, gsearch6.best_params_, gsearch6.best_score_
# opt = 0.01

#%%
param_test7 = {
 'reg_alpha':[ 5e-03, 1e-02, 5e-02]
}
gsearch7 = GridSearchCV(estimator=xgb4,param_grid=param_test7,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch7.fit(train[predictors],train[target])
gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_
# opt = 5

#%%
param_test8 = {
 'reg_alpha':[i/1000 for i in range(0,100)]
}
gsearch7 = GridSearchCV(estimator=xgb4,param_grid=param_test8,scoring='accuracy',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch7.fit(train[predictors],train[target])
gsearch7.cv_results_, gsearch7.best_params_, gsearch7.best_score_
# opt = 0.008



#%%
xgb5 = XGBClassifier(
    num_class=NUM_CLASS,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.75,
    reg_alpha=0.008,
    objective= 'multi:softprob',
    nthread=-1,
    scale_pos_weight=1,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)
modelfit(xgb5, train, predictors)

#step 6
#%%
xgb6 = XGBClassifier(
    num_class=NUM_CLASS,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.65,
    colsample_bytree=0.6,
    reg_alpha=5.6,
    objective= 'multi:softprob',
    nthread=4,
    scale_pos_weight=1,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)
for lr in [1e-5, 1e-4, 1e-3, 1e-2, 0.1]:
    xgb6.set_params(learning_rate=lr)
    xgb_param = xgb6.get_xgb_params()
    xgtrain = xgb.DMatrix(train[predictors].values, label=train[target].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=xgb6.get_params()['n_estimators'], nfold=5,
        metrics='merror', early_stopping_rounds=50, verbose_eval=False)
    nest_for_lr = cvresult.shape[0]
    print("{}: {}" .format(lr, nest_for_lr))

#%%
xgb7 = XGBClassifier(
    #num_class=NUM_CLASS,
    learning_rate=0.1,
    n_estimators=105,
    max_depth=3,
    min_child_weight=1,
    gamma=0.008,
    subsample=0.7,
    colsample_bytree=0.75,
    #reg_alpha=5.6,
    objective='binary:logistic',
    nthread=4,
    #scale_pos_weight=1,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)
param_test10 = {
    'learning_rate' : [1e-5, 1e-4, 1e-3, 1e-2, 0.1],
    'n_estimators' : [100, 500, 1000, 1500, 2000]
}
gsearch8 = GridSearchCV(estimator=xgb7,param_grid=param_test10,scoring='f1',n_jobs=4,cv=group_kfold.split(train[predictors],train[target],groups))
gsearch8.fit(train[predictors],train[target])
gsearch8.cv_results_, gsearch8.best_params_, gsearch8.best_score_

#%%
modelfit(xgb7, train, predictors)
stage_1_label = xgb7.predict(test_data[predictors])
print(stage_1_label)
#%%
test_data['Churn Category'] = stage_1_label

#%%
test_data[['Customer ID', 'Churn Category']].to_csv('./prediction/stage_1_label.csv', index=False)
