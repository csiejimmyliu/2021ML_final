#%%
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

APPLY_NORMALIZE = True

# %%
dmg = pd.read_csv("./data/demographics.csv")
loca = pd.read_csv("./data/location.csv")
pop = pd.read_csv("./data/population.csv") 
sat = pd.read_csv("./data/satisfaction.csv") 
serv = pd.read_csv("./data/services.csv") 
stat = pd.read_csv("./data/status.csv") 
test = pd.read_csv("./data/Test_IDs.csv") 
train = pd.read_csv("./data/Train_IDs.csv") 

#%%
###################
# Pre-processing 
###################

#%%
# map city and zip to population
# if nan, use average population of one city or total average
loca = loca[['Customer ID','City','Zip Code']].sort_values(['City','Zip Code'])
total_pop = 0
total_zip_cnt = 0
city_avg_pop = dict()
for city in loca['City'].unique():
    if city == np.nan:
        continue
    city_zips = loca.loc[loca['City'] == city]['Zip Code'].unique()
    city_pop = 0
    city_zip_cnt = 0
    for zip_code in city_zips:
        if zip_code == np.nan:
            continue
        zip_pop = pop.loc[pop['Zip Code'] == zip_code]['Population']
        if len(zip_pop) > 0:
            city_pop += zip_pop.iloc[0]
            city_zip_cnt += 1
    if city_zip_cnt == 0:
        city_avg_pop[city] = 0
    else:
        city_avg_pop[city] = city_pop / city_zip_cnt
    total_pop += city_pop
    total_zip_cnt += city_zip_cnt
city_avg_pop[np.nan] = total_pop / total_zip_cnt
# for key, value in city_avg_pop.items():
#     print("{}: {}" .format(key, value))

loca_pop=pd.merge(
    left=loca,
    right=pop,
    how='left',
    on='Zip Code'
)
for i in range(len(loca_pop)):
    if np.isnan(loca_pop['Population'].iloc[i]):
        city = loca_pop['City'].iloc[i]
        loca_pop['Population'].iloc[i] = int(city_avg_pop[city])

#%%
# merge different tables by Customer ID
data = pd.merge(
    left=dmg[['Customer ID','Gender','Age','Married','Number of Dependents']],
    right=loca_pop[['Customer ID','Population']],
    how='outer',
    on='Customer ID',
    suffixes=('_dmg', '_loca')
)

data = pd.merge(
    left=data,
    right=sat,
    how='outer',
    on='Customer ID',
    suffixes=('', '_sat')
)
data = pd.merge(
    left=data,
    right=serv.drop(['Count', 'Quarter', 'Referred a Friend'], axis=1),
    how='outer',
    on='Customer ID',
    suffixes=('', '_serv')
)
data = pd.merge(
    left=data,
    right=stat,
    how='outer',
    on='Customer ID',
    suffixes=('', '_stat')
)

#%%
# categorize features
cat_cols = [
    'Gender',
    'Married',
    'Offer',
    'Phone Service',
    'Multiple Lines',
    'Internet Service',
    'Internet Type',
    'Online Security',
    'Online Backup',
    'Device Protection Plan',
    'Premium Tech Support',
    'Streaming TV',
    'Streaming Movies',
    'Streaming Music',
    'Unlimited Data',
    'Contract',
    'Paperless Billing',
    'Payment Method',

]

real_cols_fill_avg = [
    'Age',
    'Population',
    'Satisfaction Score',
    'Tenure in Months',
    'Avg Monthly Long Distance Charges',
    'Avg Monthly GB Download',
    'Monthly Charge',
    'Total Charges',
    'Total Long Distance Charges',
    'Total Revenue',
]

real_cols_fill_zero = [
    'Number of Dependents',
    'Number of Referrals',
    'Total Refunds',
    'Total Extra Data Charges',
]

label_col = 'Churn Category'

#%%
# visualization
for col in real_cols_fill_avg:
    plot = data[[col]].plot(kind='hist', bins=30)
    fig = plot.get_figure()
    fig.savefig("./img/before_normalize/bn_" + col + ".png")

for col in real_cols_fill_zero:
    plot = data[[col]].plot(kind='hist', bins=30)
    fig = plot.get_figure()
    fig.savefig("./img/before_normalize/bn_" + col + ".png")

#%%
# min-max normalization
def min_max_normalize(s):
    return (s - s.min())/(s.max()-s.min())

normalized_data = data.copy()
normalized_data[real_cols_fill_avg] = normalized_data[real_cols_fill_avg].apply(min_max_normalize, axis=0)
normalized_data[real_cols_fill_zero] = normalized_data[real_cols_fill_zero].apply(min_max_normalize, axis=0)

#%%
# one hot
if (APPLY_NORMALIZE):
    onehot_data = pd.get_dummies(data=normalized_data, columns=cat_cols, dummy_na=True)
else:
    onehot_data = pd.get_dummies(data=data, columns=cat_cols, dummy_na=True)

#%%
# fill real-valued nan
for col in real_cols_fill_avg:
    onehot_data[col].fillna(onehot_data[col].mean(), inplace=True)

for col in real_cols_fill_zero:
    onehot_data[col].fillna(0, inplace=True)

#%%
# map label to numeric value
label_cat = dict({
    'No Churn' : 0,
    'Competitor' : 1,
    'Dissatisfaction' : 2,
    'Attitude' : 3,
    'Price' : 4,
    'Other' : 5,
    np.nan : -1
})
onehot_data[label_col] = onehot_data[label_col].map(label_cat)

#%%
# end of preprocessing
clean_data = onehot_data[
    [ col for col in onehot_data.columns if col != label_col ] + [label_col]
]
for col in clean_data.columns:
    nan_num = clean_data[col].isnull().sum()
    if nan_num > 0:
        print("{}: {}" .format(col, nan_num))
clean_data

#%%
# split train & test
train_data = pd.merge(
    left=train,
    right=clean_data,
    how='left',
    on='Customer ID',
    suffixes=('', '_')
)
test_data = pd.merge(
    left=test,
    right=clean_data,
    how='left',
    on='Customer ID',
    suffixes=('', '_')
).drop(label_col, axis=1)

#%%
# dump data
if (APPLY_NORMALIZE):
    train_data.to_csv('./preprocessed_data/train_data_normalized.csv', index=False)
    test_data.to_csv('./preprocessed_data/test_data_normalized.csv', index=False)
else:
    train_data.to_csv('./preprocessed_data/train_data.csv', index=False)
    test_data.to_csv('./preprocessed_data/test_data.csv', index=False)