#%%
import math
import pandas as pd
import numpy as np

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
# map city and zip to population of the city
loca = loca[['Customer ID','City','Zip Code']].sort_values(['City','Zip Code'])
for i in range(1, len(loca)):
    pre = loca.iloc[i - 1]
    cur = loca.iloc[i]
    if math.isnan(cur['Zip Code']):
        if cur['City'] == pre['City']:
            if math.isnan(pre['Zip Code']) == False:
                loca['Zip Code'].iloc[i] = loca['Zip Code'].iloc[i - 1]
loca_pop=pd.merge(
    left=loca,
    right=pop,
    how='left',
    on='Zip Code'
)

#%%
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
data

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
###################
# Pre-processing 
###################

#%%
# one hot
onehot_data = pd.get_dummies(data=data, columns=cat_cols, dummy_na=True)

#%%
# fill real-valued nan
for col in real_cols_fill_avg:
    onehot_data[col].fillna(onehot_data[col].mean(), inplace=True)

for col in real_cols_fill_zero:
    onehot_data[col].fillna(0, inplace=True)

# label to numeric category
label_cat = dict({
    'No Churn' : 0,
    'Competitor' : 1,
    'Dissatisfaction' : 2,
    'Attitude' : 3,
    'Price' : 4,
    'Other' : 5,
    np.nan : -1
})
onehot_data[label_col] = onehot_data[label_col].map(label_cat).value_counts()

#%%
# todo: feature selection


#%%
train_data = pd.merge(
    left=train,
    right=data,
    how='left',
    on='Customer ID',
    suffixes=('', '_')
)
test_data = pd.merge(
    left=test,
    right=data,
    how='left',
    on='Customer ID',
    suffixes=('', '_')
)

#%%
# todo: modeling


#%%
loca.loc[loca['City'] == 'Los Angeles']

#%%
len(data)

#%%
len(train_data)

#%%
train_data["Churn Category"].isnull().sum()

#%%
len(test_data)

#%%
test_data["Churn Category"].isnull().sum()

#%%
for col in train_data.columns:
    nan_num = train_data[col].isnull().sum()
    print("{}: {}" .format(col, nan_num))

#%%
train_data[["Customer ID", "Churn Category"]]
