#%%
import pandas as pd

#%%
stage_1 = pd.read_csv('./prediction/stage_1_label.csv')
stage_2 = pd.read_csv('./prediction/stage_2_label.csv')
test = pd.read_csv('./data/Test_IDs.csv')

#%%
test_result = pd.merge(
    left=test,
    right=stage_1.loc[stage_1['Churn Category'] == 0],
    how='left',
    on='Customer ID',
    suffixes=('', '_1')
)

test_result = pd.merge(
    left=test_result,
    right=stage_2,
    how='left',
    on='Customer ID',
    suffixes=('', '_2')
)

#%%
for i in range(len(test_result)):
    if pd.isnull(test_result['Churn Category'].iloc[i]):
        test_result['Churn Category'].iloc[i] = test_result['Churn Category_2'].iloc[i]


test_result = test_result.drop('Churn Category_2', axis=1)
test_result['Churn Category'] = test_result['Churn Category'].astype(int)
test_result.to_csv('./prediction/version2.csv', index=False)