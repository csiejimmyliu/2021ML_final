#%%
# Import libraries
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
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics   #Additional scklearn functions

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
from sklearn.metrics import precision_score, make_scorer

rcParams['figure.figsize'] = 12, 4

#%%
train = pd.read_csv('./preprocessed_data/train_data_normalized.csv')
train_no_label = train.loc[train['Churn Category'] == -1]
train = train.loc[train['Churn Category'] != -1]
test_data = pd.read_csv('./preprocessed_data/test_data_normalized.csv')

target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]

#%%
num_list=[]

for i in range(6):
    num_list.append(sum(train['Churn Category']==i))

N = num_list[0] / 5

for i in range(1,6):
    # print(train.loc[train['Churn Category'] == i].sample(n=int(N-num_list[i]),replace=True))
    train=train.append(train.loc[train['Churn Category'] == i].sample(n=int(N-num_list[i]),replace=True))

#%%
num_list=[]

for i in range(6):
    num_list.append(sum(train['Churn Category']==i))

num_list

#%%
train[target]=np.where(train[target]==0,0,1)
train = train.reset_index(drop=True)

#%%
def modelfit(alg, dtrain, predictors,useTrainCV=False,cv_folds=5, early_stopping_rounds=50):
    
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
xgb_model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=105,
    max_depth=13,
    min_child_weight=0,
    gamma=0,
    subsample=0.7,
    colsample_bytree=0.7,
    objective='binary:logistic',
    nthread=4,
    seed=1126,
    verbosity=0, 
    use_label_encoder=False
)

# modelfit(xgb_model, train, predictors)
#%%
X = train[predictors].copy()
y = train[[target]].copy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1126)
validation_results = []
for train_index, val_index in skf.split(X, y):
    # print("TRAIN:", train_index, "VAL:", val_index)
    X_train, X_val = X.loc[train_index], X.loc[val_index]
    y_train, y_val = y.loc[train_index], y.loc[val_index]

    # Fit the algorithm on the data
    xgb_model.fit(X_train, y_train, eval_metric='error')
        
    # Predict validation set:
    y_val['Predicted_Churn_Category'] = xgb_model.predict(X_val)
    y_val[['Class_0_Prob','Class_1_Prob']] = xgb_model.predict_proba(X_val)
    validation_results.append(y_val)

#%%
for i in range(len(validation_results)):
    val_result = validation_results[i]
    val_result['Prob'] = val_result[['Class_0_Prob','Class_1_Prob']].max(axis=1)
    correct = val_result.loc[(val_result['Predicted_Churn_Category'] == val_result[target]) & (val_result['Prob'] < 0.9)]
    wrong = val_result.loc[(val_result['Predicted_Churn_Category'] != val_result[target]) & (val_result['Prob'] < 0.9)]
    bins = np.linspace(0.5, 1, 50)
    plt.hist(correct['Prob'], bins, alpha=0.5, label='correct ' + str(i + 1))
    plt.xticks(np.arange(0.5, 1, 0.05))
    plt.legend(loc='upper right')
    plt.show()
    plt.hist(wrong['Prob'], bins, alpha=0.5, label='wrong ' + str(i + 1))
    plt.xticks(np.arange(0.5, 1, 0.05))
    plt.legend(loc='upper right')
    plt.show()

#%%

#%%
train['Predicted_Churn_Category'] = xgb.predict(train[train_no_label])

#%%
train[['Class_0_Prob','Class_1_Prob']] = xgb.predict_proba(train[predictors])

#%%
correct = train.loc[train['Predtict Churn Category'] == train[target]][ID_]
wrong = train.loc[train['Predtict Churn Category'] != train[target]]

#%%
correct

#%%
test_data['Churn Category'] = stage_1_label

#%%
test_data[['Customer ID', 'Churn Category']].to_csv('./prediction/stage_1_label.csv', index=False)

#%%
################################
# Pseudo-Label
################################

#%%
train = pd.read_csv('./preprocessed_data/train_data_normalized.csv')
train_no_label = train.loc[train['Churn Category'] == -1]
train = train.loc[train['Churn Category'] != -1]
test_data = pd.read_csv('./preprocessed_data/test_data_normalized.csv')

target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in train.columns if x not in [target, IDcol]]

#%%
train[target] = np.where(train[target]==0, 0, 1)
train = train.reset_index(drop=True)

#%%
xgb_model = XGBClassifier()
xgb_model.load_model('./stage_1_v1.json')

#%%
# prediction accuracy on labeled-train
train_predictions = xgb_model.predict(train[predictors])
print("Accuracy : %.4g" % metrics.accuracy_score(train[target].values, train_predictions))

#%%
train_no_label[target] = xgb_model.predict(train_no_label[predictors])
train_add_pseudo = train.append(train_no_label, ignore_index=True)

#%%
xgb_model.fit(train_add_pseudo[predictors], train_add_pseudo[target], eval_metric='error')

#%%
train_predictions = xgb_model.predict(train[predictors])
print("Accuracy : %.4g" % metrics.accuracy_score(train[target].values, train_predictions))

#%%
xgb_model.save_model('./stage_1_pseudo_label_v1.json')

# %%
