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

#%%
# todo: some pre-process like yes/no replacement, outlier, nan, ...


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

#%%
