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
def list_nan(df):
    for col in df.columns:
        print("nan in {}: {} / {}" .format(col, df[col].isnull().sum(), len(df)))

#%%
###################
# Pre-processing Start
###################

#%%
###################
# demographics.csv 
###################
list_nan(dmg)

#%%
# Under 30, Senior Citizen -> Age
avg_lt_30 = round(dmg.loc[dmg['Age'] < 30]['Age'].mean())
avg_ge30_lt65 = round(dmg.loc[(dmg['Age'] >= 30) & (dmg['Age'] < 65)]['Age'].mean())
avg_ge_65 = round(dmg.loc[dmg['Age'] >= 65]['Age'].mean())

def reco_age(age, under_30, senior):
    if np.isnan(age) == False:
        return age
    if under_30 == 'Yes':
        return avg_lt_30
    if senior == 'Yes':
        return avg_ge_65
    if under_30 == 'No' and senior == 'No':
        return avg_ge30_lt65
    return np.nan

dmg['Age'] = dmg.apply(lambda x: reco_age(x['Age'], x['Under 30'], x['Senior Citizen']), axis=1)

#%%
# Dependents -> Number of Dependents
avg_dep = round(dmg.loc[dmg['Dependents'] == 'Yes']['Number of Dependents'].mean())

def reco_num_dep(dep, num_dep):
    if np.isnan(num_dep) == False:
        return num_dep
    if dep == 'Yes':
        return avg_dep
    if dep == 'No':
        return 0
    return np.nan

dmg['Number of Dependents'] = dmg.apply(lambda x: reco_num_dep(x['Dependents'], x['Number of Dependents']), axis=1)

#%%
###################
# location.csv 
###################
list_nan(loca)

#%%
# Latitude -> Longitude, Longitude -> Latitude
lat_to_lon = loca[['Latitude', 'Longitude']].dropna().drop_duplicates().set_index('Latitude').to_dict()['Longitude']
lon_to_lat = loca[['Latitude', 'Longitude']].dropna().drop_duplicates().set_index('Longitude').to_dict()['Latitude']

def reco_lat(lat, lon):
    if np.isnan(lat) == False:
        return lat
    if np.isnan(lon) == False:
        return lon_to_lat.get(lon, np.nan)
    return np.nan

loca['Latitude'] = loca.apply(lambda x: reco_lat(x['Latitude'], x['Longitude']), axis=1)

def reco_lon(lat, lon):
    if np.isnan(lon) == False:
        return lon
    if np.isnan(lat) == False:
        return lat_to_lon.get(lat, np.nan)
    return np.nan

loca['Longitude'] = loca.apply(lambda x: reco_lon(x['Latitude'], x['Longitude']), axis=1)

#%%
# Latitude, Longitude -> Lat Long
def reco_lat_lon(lat_lon, lat, lon):
    if pd.isnull(lat_lon) == False:
        return lat_lon
    if np.isnan(lat) == False and np.isnan(lon) == False:
        return str(lat) + ", " + str(lon)
    return np.nan

loca['Lat Long'] = loca.apply(lambda x: reco_lat_lon(x['Lat Long'], x['Latitude'], x['Longitude']), axis=1)

#%%
# Lat Long -> Latitude, Longitude
loca[['Latitude','Longitude']] = loca['Lat Long'].str.split(', ', 1, expand=True)

#%%
# Lat Long -> Zip Code
latlon_to_zip = loca[['Zip Code', 'Lat Long']].dropna().drop_duplicates().set_index('Lat Long').to_dict()['Zip Code']

def reco_zip(zip_code, lat_lon):
    if np.isnan(zip_code) == False:
        return zip_code
    if pd.isnull(lat_lon) == False:
        return latlon_to_zip.get(lat_lon, np.nan)

loca['Zip Code'] = loca.apply(lambda x: reco_zip(x['Zip Code'], x['Lat Long']), axis=1)

#%%
# City, Zip Code -> Population
# if Zip Code is nan, use average Population of City
# if Zip Code and City is both nan, use average Population of all Cities
loca = loca[['Customer ID','City','Zip Code','Latitude','Longitude']].sort_values(['City','Zip Code'])
total_pop = 0
total_zip_cnt = 0
city_avg_pop = dict()
for city in loca['City'].unique():
    if pd.isnull(city):
        continue
    city_zips = loca.loc[loca['City'] == city]['Zip Code'].unique()
    city_pop = 0
    city_zip_cnt = 0
    for zip_code in city_zips:
        if pd.isnull(zip_code):
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
        loca_pop.iloc[i, loca_pop.columns.get_loc('Population')] = round(city_avg_pop[city])

#%%
###################
# services.csv 
###################
list_nan(serv)

#%%
# Referred a Friend -> Number of Referrals
avg_yes_num_refer = round(serv.loc[serv['Referred a Friend'] == 'Yes']['Number of Referrals'].mean())

def reco_num_refer(ref, num_ref):
    if np.isnan(num_ref) == False:
        return num_ref
    if ref == 'Yes':
        return avg_yes_num_refer
    if ref == 'No':
        return 0
    return np.nan

serv['Number of Referrals'] = serv.apply(lambda x: reco_num_refer(x['Referred a Friend'], x['Number of Referrals']), axis=1)

#%%
# Number of Referrals -> Referred a Friend
def reco_refer(ref, num_ref):
    if pd.isnull(ref) == False:
        return ref
    if num_ref > 0:
        return 'Yes'
    if num_ref == 0:
        return 'No'
    return np.nan

serv['Referred a Friend'] = serv.apply(lambda x: reco_refer(x['Referred a Friend'], x['Number of Referrals']), axis=1)

#%%
# Total Charges, Monthly Charge -> Tenure in Months
def reco_tenure_by_normal_charge(tenure, ttl_charge, mon_charge):
    if np.isnan(tenure) == False:
        return tenure
    if np.isnan(ttl_charge) == False and np.isnan(mon_charge) == False and mon_charge != 0:
        return math.ceil(ttl_charge / mon_charge)
    return np.nan

serv['Tenure in Months'] = serv.apply(lambda x: reco_tenure_by_normal_charge(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)

#%%
# Tenure in Months, Monthly Charge -> Total Charges
def reco_ttl_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(ttl_charge) == False:
        return ttl_charge
    if np.isnan(tenure) == False and np.isnan(mon_charge) == False:
        return tenure * mon_charge
    return np.nan

serv['Total Charges'] = serv.apply(lambda x: reco_ttl_charge_by_tenure(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)

#%%
# Tenure in Months, Total Charges -> Monthly Charge
def reco_mon_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(mon_charge) == False:
        return mon_charge
    if np.isnan(tenure) == False and np.isnan(ttl_charge) == False and tenure != 0:
        return ttl_charge / tenure
    return np.nan

serv['Monthly Charge'] = serv.apply(lambda x: reco_mon_charge_by_tenure(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)

#%%
# Total Long Distance Charges, Avg Monthly Long Distance Charges -> Tenure in Months
def reco_tenure_by_long_charge(tenure, ttl_charge, mon_charge):
    if np.isnan(tenure) == False:
        return tenure
    if np.isnan(ttl_charge) == False and np.isnan(mon_charge) == False and mon_charge != 0:
        return math.ceil(ttl_charge / mon_charge)
    return np.nan

serv['Tenure in Months'] = serv.apply(lambda x: reco_tenure_by_long_charge(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)

#%%
# Tenure in Months, Avg Monthly Long Distance Charges -> Total Long Distance Charges
def reco_long_ttl_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(ttl_charge) == False:
        return ttl_charge
    if np.isnan(tenure) == False and np.isnan(mon_charge) == False:
        return tenure * mon_charge
    return np.nan

serv['Total Long Distance Charges'] = serv.apply(lambda x: reco_long_ttl_charge_by_tenure(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)

#%%
# Tenure in Months, Total Long Distance Charges -> Avg Monthly Long Distance Charges
def reco_long_mon_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(mon_charge) == False:
        return mon_charge
    if np.isnan(tenure) == False and np.isnan(ttl_charge) == False and tenure != 0:
        return ttl_charge / tenure
    return np.nan

serv['Avg Monthly Long Distance Charges'] = serv.apply(lambda x: reco_long_mon_charge_by_tenure(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)

#%%
# Total Charges, Total Refunds, Total Extra Data Charges, Total Long Distance Charges -> Total Revenue
def reco_revenue(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(revenue) == False:
        return revenue
    if np.isnan(ttl_charge) == False and np.isnan(refund) == False and np.isnan(data_charge) == False and np.isnan(long_charge) == False:
        return ttl_charge - refund + data_charge + long_charge
    return np.nan

serv['Total Revenue'] = serv.apply(lambda x: reco_revenue(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

#%%
# Total Revenue, Total Refunds, Total Extra Data Charges, Total Long Distance Charges -> Total Charges
def reco_ttl_charge(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(ttl_charge) == False:
        return ttl_charge
    if np.isnan(revenue) == False and np.isnan(refund) == False and np.isnan(data_charge) == False and np.isnan(long_charge) == False:
        return revenue + refund - data_charge - long_charge
    return np.nan

serv['Total Charges'] = serv.apply(lambda x: reco_ttl_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

#%%
# Total Revenue, Total Charges, Total Extra Data Charges, Total Long Distance Charges -> Total Refunds
def reco_refund(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(refund) == False:
        return refund
    if np.isnan(revenue) == False and np.isnan(ttl_charge) == False and np.isnan(data_charge) == False and np.isnan(long_charge) == False:
        return ttl_charge + data_charge + long_charge - revenue
    return np.nan

serv['Total Refunds'] = serv.apply(lambda x: reco_refund(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

#%%
# Total Revenue, Total Charges, Total Refunds, Total Long Distance Charges -> Total Extra Data Charges
def reco_data_charge(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(data_charge) == False:
        return data_charge
    if np.isnan(revenue) == False and np.isnan(ttl_charge) == False and np.isnan(refund) == False and np.isnan(long_charge) == False:
        return revenue - ttl_charge + refund - long_charge
    return np.nan

serv['Total Extra Data Charges'] = serv.apply(lambda x: reco_data_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

#%%
# Total Revenue, Total Charges, Total Refunds, Total Extra Data Charges -> Total Long Distance Charges
def reco_long_charge(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(long_charge) == False:
        return long_charge
    if np.isnan(revenue) == False and np.isnan(ttl_charge) == False and np.isnan(refund) == False and np.isnan(data_charge) == False:
        return revenue - ttl_charge + refund - data_charge
    return np.nan

serv['Total Long Distance Charges'] = serv.apply(lambda x: reco_long_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

#%%
# Tenure in Months = Total Charges / Monthly Charge 
# Tenure in Months = Total Long Distance Charges / Avg Monthly Long Distance Charges
# Total Revenue = Total Charges - Total Refunds + Total Extra Data Charges + Total Long Distance Charges
cycle = [
    'Tenure in Months',
    'Total Charges',
    'Monthly Charge',
    'Total Long Distance Charges',
    'Avg Monthly Long Distance Charges',
    'Total Refunds',
    'Total Extra Data Charges',
    'Total Revenue'
]
iter = 1
while True:
    print("iter: {}" .format(iter))
    nan_before = 0
    for col in cycle:
        nan_before += serv[col].isnull().sum()

    serv['Tenure in Months'] = serv.apply(lambda x: reco_tenure_by_normal_charge(
        x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
    )
    serv['Total Charges'] = serv.apply(lambda x: reco_ttl_charge_by_tenure(
        x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
    )
    serv['Monthly Charge'] = serv.apply(lambda x: reco_mon_charge_by_tenure(
        x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
    )
    serv['Tenure in Months'] = serv.apply(lambda x: reco_tenure_by_long_charge(
        x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
    )
    serv['Total Long Distance Charges'] = serv.apply(lambda x: reco_long_ttl_charge_by_tenure(
        x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
    )
    serv['Avg Monthly Long Distance Charges'] = serv.apply(lambda x: reco_long_mon_charge_by_tenure(
        x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
    )
    serv['Total Revenue'] = serv.apply(lambda x: reco_revenue(
        x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
    )
    serv['Total Charges'] = serv.apply(lambda x: reco_ttl_charge(
        x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
    )
    serv['Total Refunds'] = serv.apply(lambda x: reco_refund(
        x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
    )
    serv['Total Extra Data Charges'] = serv.apply(lambda x: reco_data_charge(
        x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
    )
    serv['Total Long Distance Charges'] = serv.apply(lambda x: reco_long_charge(
        x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
    )

    iter += 1
    nan_after = 0
    for col in cycle:
        nan_after += serv[col].isnull().sum()
    if nan_before == nan_after:
        break

#%%
# Multiple Lines -> Phone Service
def reco_phone(phone, multi):
    if pd.isnull(phone) == False:
        return phone
    if multi == 'Yes':
        return 'Yes'
    return np.nan

serv['Phone Service'] = serv.apply(lambda x: reco_phone(
    x['Phone Service'], x['Multiple Lines']), axis=1
)

#%%
# Phone Service -> Multiple Lines
def reco_multi(phone, multi):
    if pd.isnull(multi) == False:
        return multi
    if phone == 'No':
        return 'No'
    return np.nan

serv['Multiple Lines'] = serv.apply(lambda x: reco_multi(
    x['Phone Service'], x['Multiple Lines']), axis=1
)

#%%
# Internet Type -> Internet Service
def reco_internet_from_internet_type(internet, internet_type):
    if pd.isnull(internet) == False:
        return internet
    if internet_type in ['Cable','DSL','Fiber Optic']:
        return 'Yes'
    return np.nan

serv['Internet Service'] = serv.apply(lambda x: reco_internet_from_internet_type(
    x['Internet Service'], x['Internet Type']), axis=1
)

# Avg Monthly GB Download -> Internet Service
def reco_internet_from_internet_type(internet, gb):
    if pd.isnull(internet) == False:
        return internet
    if gb > 0:
        return 'Yes'
    return np.nan

serv['Internet Service'] = serv.apply(lambda x: reco_internet_from_internet_type(
    x['Internet Service'], x['Avg Monthly GB Download']), axis=1
)

# reconstruct Internet Service
def reco_internet(internet, constructor):
    if pd.isnull(internet) == False:
        return internet
    if constructor == 'Yes':
        return 'Yes'
    return np.nan

for col in ['Online Security','Online Backup',
            'Device Protection Plan','Premium Tech Support','Streaming TV',
            'Streaming Movies','Streaming Music','Unlimited Data']:
    serv['Internet Service'] = serv.apply(lambda x: reco_internet(
        x['Internet Service'], x[col]), axis=1
    )

# Internet Service -> Internet Type
def reco_internet_type(internet, internet_type):
    if pd.isnull(internet_type) == False:
        return internet_type
    if internet == 'No':
        return 'None'
    return np.nan

serv['Internet Type'] = serv.apply(lambda x: reco_internet_type(
    x['Internet Service'], x['Internet Type']), axis=1
)

# Internet Service -> Avg Monthly GB Download
def reco_GB(internet, gb):
    if pd.isnull(gb) == False:
        return gb
    if internet == 'No':
        return 0
    return np.nan

serv['Avg Monthly GB Download'] = serv.apply(lambda x: reco_GB(
    x['Internet Service'], x['Avg Monthly GB Download']), axis=1
)

# reconstrcut from Internet Service
def reco_from_internet(internet, to_construct):
    if pd.isnull(to_construct) == False:
        return to_construct
    if internet == 'No':
        return 'No'
    return np.nan

for col in ['Online Security','Online Backup','Device Protection Plan',
            'Premium Tech Support','Streaming TV',
            'Streaming Movies','Streaming Music','Unlimited Data']:
    serv['Internet Service'] = serv.apply(lambda x: reco_from_internet(
        x['Internet Service'], x[col]), axis=1
    )

#%%
# merge different tables by Customer ID
data = pd.merge(
    left=dmg[['Customer ID','Gender','Age','Married','Number of Dependents']],
    right=loca_pop[['Customer ID','Population','Latitude','Longitude']],
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

real_cols_fill_mode = [
    'Latitude',
    'Longitude'
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
normalized_data[real_cols_fill_mode] = normalized_data[real_cols_fill_mode].astype(float).apply(min_max_normalize, axis=0)

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

for col in real_cols_fill_mode:
    onehot_data[col].fillna(onehot_data[col].mode()[0], inplace=True)

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
###################
# Pre-processing End
###################
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