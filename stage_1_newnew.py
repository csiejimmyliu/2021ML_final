#%%
# import library
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
from sklearn import metrics   #Additional scklearn functions
from sklearn.model_selection import StratifiedGroupKFold,StratifiedKFold

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
SEED_list=[i for i in range(0,15)]

#%%
# import data
if (APPLY_NORMALIZATION):
    total_train = pd.read_csv('./preprocessed_data/train_data_std_normalized.csv')
    total_train = total_train.loc[total_train['Churn Category'] != -1]
    test_data = pd.read_csv('./preprocessed_data/test_data_std_normalized.csv')
else:
    total_train = pd.read_csv('./preprocessed_data/train_data.csv')
    total_train = total_train.loc[total_train['Churn Category'] != -1]
    test_data = pd.read_csv('./preprocessed_data/test_data.csv')

target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in total_train.columns if x not in [target, IDcol]]

#%%
total_train[target].value_counts()

#%%
for turn in range(len(SEED_list)):
    temp_train=total_train.copy()
    temp_train['Group Label'] = np.array(list(range(len(temp_train))))
    train_0=temp_train.loc[temp_train['Churn Category'] == 0]
    train_0=train_0.sample(n=600,replace=False,random_state=SEED_list[turn])
    
    num_list=[]
    for i in range(6):
        num_list.append(sum(temp_train['Churn Category']==i))


    N=len(train_0)/5
    train=train_0

    
    


    for i in range(1,6):
        train=train.append(temp_train.loc[temp_train['Churn Category'] == i].sample(n=120,replace=True,random_state=SEED_list[turn]))

    
    
    groups = np.array(train['Group Label'])
    train.drop('Group Label', axis=1, inplace=True)
    train['Churn Category']=np.where(train[target]==0,0,1)


    
    '''
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED_list[turn])
    train, val = next(kfold.split(train[predictors], train[target]))
    X_sample = X.iloc[val, :]
    y_sample = y.iloc[val]
    y_sample.value_counts()
    '''


    '''
    # drop category 0, category No. minus 1
    train_1to5 = train.loc[train['Churn Category']  != 0].copy()
    for i in range(1,6):
        train_1to5['Churn Category'].replace(i, i-1, inplace=True)
    if WITH_GROUPING:
        train_1to5['Group Label'] = np.array(list(range(len(train_1to5))))
    train_1to5['Churn Category'].value_counts()


    # random oversample and grouping
    # prevent duplicate examples from appearing in both training and validation sets
    oversample = RandomOverSampler(random_state=SEED_list[turn])
    if WITH_GROUPING:
        X = train_1to5[predictors+['Group Label']]
    else:
        X = train_1to5[predictors]
    y = train_1to5[target]
    X_res, y_res = oversample.fit_resample(X, y)
    print(y_res.value_counts())
    '''


    if WITH_GROUPING:
        
        group_kfold = StratifiedGroupKFold(n_splits=5, shuffle=False)
    

    X_res=train[predictors]
    y_res=train[target]
    # xgboost hyperparameters
    TUNING_STAGES = 5
    initial_params = {
        
        'learning_rate': 0.1,
        'n_estimators': 153,
        'max_depth': 10,
        'min_child_weight': 1.0,
        'gamma': 0.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'nthread': 4,
        'seed': SEED_list[turn],
        'verbosity': 0,
        'use_label_encoder': False,
        'reg_alpha': 0
    }
    param_iterations = [initial_params] * (TUNING_STAGES + 1)
    scores = [0] * (TUNING_STAGES + 1)


    # cross validate model
    def cv_model(params, data_X, data_y, folds):
        alg = XGBClassifier(
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
        cv_results = cross_validate(alg, data_X, data_y, 
            cv=folds.split(data_X, data_y, groups), scoring='f1')
        return cv_results['test_score']


    # grid search function
    def grid_search(original_params, data_X, data_y, param_test, folds):
        alg = XGBClassifier(
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
            use_label_encoder=original_params['use_label_encoder'],
            reg_alpha=original_params['reg_alpha']
        )
        gsearch = GridSearchCV(estimator=alg, param_grid = param_test, 
            scoring='f1_micro',n_jobs=16, cv=folds.split(data_X, data_y, groups))
        gsearch.fit(data_X, data_y)
        return gsearch.best_score_, gsearch.best_params_


  

    # grid search 1
    iter_num = 1
    param_test1 = {
        'min_child_weight':range(1, 3, 2)
    }
   
    scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, param_test1, group_kfold)
    
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()


    # grid search 2
    iter_num = 2
    prev_mcw = param_iterations[iter_num-1]['min_child_weight']
    param_test2 = {
        'min_child_weight': [
            prev_mcw-1.5, 
            prev_mcw-1, 
            prev_mcw-0.5, 
            prev_mcw, 
            prev_mcw+0.5, 
            prev_mcw+1,
            prev_mcw+1.5,
        ]
    }
    while True:
        scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
            param_test2, group_kfold)
        print('best score is ', scores[iter_num])
        if scores[iter_num] < scores[iter_num - 1]:
            print('score dropped')
            continue
        print('score better')
        break
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()


    # grid search 3
    iter_num = 3
    param_test3 = {
        'gamma':[i/10.0 for i in range(0,5)]
    }
    while True:
        scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
            param_test3, group_kfold)
        print('best score is ', scores[iter_num])
        if scores[iter_num] < scores[iter_num - 1]:
            print('score dropped')
            continue
        print('score better')
        break
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()


    # grid search 4
    iter_num = 4
    param_test4 = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
    }
    while True:
        scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
            param_test4, group_kfold)
        print('best score is ', scores[iter_num])
        if scores[iter_num] < scores[iter_num - 1]:
            print('score dropped')
            continue
        print('score better')
        break
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()


    # grid search 5
    iter_num = 5
    prev_sub = int(param_iterations[iter_num-1]['subsample'] * 100)
    prev_csb = int(param_iterations[iter_num-1]['colsample_bytree'] * 100)
    param_test5 = {
        'subsample':[i/100.0 for i in range(prev_sub-10,prev_sub+15,5)],
        'colsample_bytree':[i/100.0 for i in range(prev_csb-10,prev_csb+15,5)]
    }
    while True:
        scores[iter_num], to_update = grid_search(param_iterations[iter_num-1], X_res, y_res, 
            param_test5, group_kfold)
        print('best score is ', scores[iter_num])
        if scores[iter_num] < scores[iter_num - 1]:
            print('score dropped')
            continue
        print('score better')
        break
    new_params = param_iterations[iter_num-1].copy()
    for key, value in to_update.items():
        new_params[key] = value
    print('best params: ', to_update, '\n' ,new_params)
    param_iterations[iter_num] = new_params.copy()




    # model fit
    def get_model(params):
        alg = XGBClassifier(
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
    stage_2_model = get_model(param_iterations[iter_num])
    stage_2_model.fit(X_res, y_res, eval_metric='merror')
    feat_imp = pd.Series(stage_2_model.get_booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


    predictions = stage_2_model.predict(X_res[predictors])
    acc_score = metrics.accuracy_score(y_res.values, predictions)
    f1_score = metrics.f1_score(y_res.values, predictions, average='macro')
    print( "accuracy : %.4g" % acc_score)
    print( "f1 score : %.4g" % f1_score)
    cv_result = cv_model(param_iterations[iter_num], X_res, y_res, group_kfold)
    print("CV score : ", np.average(cv_result))


    # save_model
    stage_2_model.save_model(f'./stage_1_en_new/stage_1_model_{turn}.json')



# %%
