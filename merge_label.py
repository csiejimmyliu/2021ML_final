#%%
import pandas as pd

import xgboost as xgb
from xgboost.sklearn import XGBClassifier

#%%
test = pd.read_csv('./preprocessed_data/test_data_normalized.csv')
target = 'Churn Category'
IDcol = 'Customer ID'
predictors = [x for x in test.columns if x not in [target, IDcol]]

#%%
xgb_model_s1 = XGBClassifier()
xgb_model_s1.load_model('./stage_1_v1.json')

xgb_model_s2 = XGBClassifier()
xgb_model_s2.load_model('./stage_2_v1.json')

#%%
test[target] = xgb_model_s1.predict(test[predictors])

#%%
test_s2 = test.loc[test[target] == 1]

#%%
test_s2[target] = xgb_model_s2.predict(test_s2[predictors])
test_s2[target] = test_s2[target] + 1

#%%
test_result = pd.merge(
    left=test,
    right=test_s2[[IDcol,target]],
    how='left',
    on='Customer ID',
    suffixes=('', '_2')
)

# test_result = pd.merge(
#     left=test_result,
#     right=stage_2,
#     how='left',
#     on='Customer ID',
#     suffixes=('', '_2')
# )

#%%
for i in range(len(test_result)):
    if test_result['Churn Category'].iloc[i] == 1:
        test_result['Churn Category'].iloc[i] = test_result['Churn Category_2'].iloc[i]

#%%
test_result[target].value_counts()

#%%
test_result['Churn Category'] = test_result['Churn Category'].astype(int)
test_result[[IDcol, target]].to_csv('./prediction/version2.csv', index=False)

# %%
