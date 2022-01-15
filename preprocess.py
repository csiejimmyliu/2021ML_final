#%%
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pyplot import rcParams
rcParams['figure.figsize'] = 12, 4

APPLY_NORMALIZE = True
STD_NORMALIZE = True

# %%
dmg = pd.read_csv("./data/demographics.csv").drop('Count', axis=1)
loca = pd.read_csv("./data/location.csv").drop('Count', axis=1)
pop = pd.read_csv("./data/population.csv").drop('ID', axis=1)
serv = pd.read_csv("./data/services.csv").drop(['Count','Quarter'], axis=1)
sat = pd.read_csv("./data/satisfaction.csv") 
stat = pd.read_csv("./data/status.csv") 
train_id = pd.read_csv("./data/Train_IDs.csv")
test_id = pd.read_csv("./data/Test_IDs.csv") 

#%%
def list_nan(df):
    for col in df.columns:
        print("nan in {}: {} / {}" .format(col, df[col].isnull().sum(), len(df)))

def merge_all_tables(df, tables):
    for table in tables:
        df = pd.merge(
            left=df,
            right=table,
            how='left',
            on='Customer ID'
        )

    return df

#%%
train = merge_all_tables(train_id, [dmg, loca, serv, sat, stat])
test = merge_all_tables(test_id, [dmg, loca, serv, sat, stat])

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
avg_lt_30 = round(train.loc[train['Age'] < 30]['Age'].mean())
avg_ge30_lt65 = round(train.loc[(train['Age'] >= 30) & (train['Age'] < 65)]['Age'].mean())
avg_ge_65 = round(train.loc[train['Age'] >= 65]['Age'].mean())

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

train['Age'] = train.apply(lambda x: reco_age(x['Age'], x['Under 30'], x['Senior Citizen']), axis=1)
test['Age'] = test.apply(lambda x: reco_age(x['Age'], x['Under 30'], x['Senior Citizen']), axis=1)

#%%
# Dependents -> Number of Dependents
avg_dep = round(train.loc[train['Dependents'] == 'Yes']['Number of Dependents'].mean())

def reco_num_dep(dep, num_dep):
    if np.isnan(num_dep) == False:
        return num_dep
    if dep == 'Yes':
        return avg_dep
    if dep == 'No':
        return 0
    return np.nan

train['Number of Dependents'] = train.apply(lambda x: reco_num_dep(x['Dependents'], x['Number of Dependents']), axis=1)
test['Number of Dependents'] = test.apply(lambda x: reco_num_dep(x['Dependents'], x['Number of Dependents']), axis=1)

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
train['Latitude'] = train.apply(lambda x: reco_lat(x['Latitude'], x['Longitude']), axis=1)
test['Latitude'] = test.apply(lambda x: reco_lat(x['Latitude'], x['Longitude']), axis=1)

def reco_lon(lat, lon):
    if np.isnan(lon) == False:
        return lon
    if np.isnan(lat) == False:
        return lat_to_lon.get(lat, np.nan)
    return np.nan

loca['Longitude'] = loca.apply(lambda x: reco_lon(x['Latitude'], x['Longitude']), axis=1)
train['Longitude'] = train.apply(lambda x: reco_lon(x['Latitude'], x['Longitude']), axis=1)
test['Longitude'] = test.apply(lambda x: reco_lon(x['Latitude'], x['Longitude']), axis=1)

#%%
# Latitude, Longitude -> Lat Long
def reco_lat_lon(lat_lon, lat, lon):
    if pd.isnull(lat_lon) == False:
        return lat_lon
    if np.isnan(lat) == False and np.isnan(lon) == False:
        return str(lat) + ", " + str(lon)
    return np.nan

loca['Lat Long'] = loca.apply(lambda x: reco_lat_lon(x['Lat Long'], x['Latitude'], x['Longitude']), axis=1)
train['Lat Long'] = train.apply(lambda x: reco_lat_lon(x['Lat Long'], x['Latitude'], x['Longitude']), axis=1)
test['Lat Long'] = test.apply(lambda x: reco_lat_lon(x['Lat Long'], x['Latitude'], x['Longitude']), axis=1)

# Lat Long -> Latitude, Longitude
loca[['Latitude','Longitude']] = loca['Lat Long'].str.split(', ', 1, expand=True)
train[['Latitude','Longitude']] = train['Lat Long'].str.split(', ', 1, expand=True)
test[['Latitude','Longitude']] = test['Lat Long'].str.split(', ', 1, expand=True)

#%%
# Lat Long -> Zip Code
latlon_to_zip = loca[['Zip Code', 'Lat Long']].dropna().drop_duplicates().set_index('Lat Long').to_dict()['Zip Code']

def reco_zip(zip_code, lat_lon):
    if np.isnan(zip_code) == False:
        return zip_code
    if pd.isnull(lat_lon) == False:
        return latlon_to_zip.get(lat_lon, np.nan)

loca['Zip Code'] = loca.apply(lambda x: reco_zip(x['Zip Code'], x['Lat Long']), axis=1)
train['Zip Code'] = train.apply(lambda x: reco_zip(x['Zip Code'], x['Lat Long']), axis=1)
test['Zip Code'] = test.apply(lambda x: reco_zip(x['Zip Code'], x['Lat Long']), axis=1)

#%%
# City, Zip Code -> Population
# if Zip Code is nan, use average Population of City
# if Zip Code and City is both nan, use average Population of all Cities
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

train = pd.merge(
    left=train,
    right=pop,
    how='left',
    on='Zip Code'
)
for i in range(len(train)):
    if np.isnan(train['Population'].iloc[i]):
        city = train['City'].iloc[i]
        train.iloc[i, train.columns.get_loc('Population')] = round(city_avg_pop[city])

test = pd.merge(
    left=test,
    right=pop,
    how='left',
    on='Zip Code'
)
for i in range(len(test)):
    if np.isnan(test['Population'].iloc[i]):
        city = test['City'].iloc[i]
        test.iloc[i, test.columns.get_loc('Population')] = round(city_avg_pop[city])

#%%
###################
# services.csv 
###################
list_nan(serv)

#%%
# Referred a Friend -> Number of Referrals
avg_yes_num_refer = round(train.loc[train['Referred a Friend'] == 'Yes']['Number of Referrals'].mean())

def reco_num_refer(ref, num_ref):
    if np.isnan(num_ref) == False:
        return num_ref
    if ref == 'Yes':
        return avg_yes_num_refer
    if ref == 'No':
        return 0
    return np.nan

train['Number of Referrals'] = train.apply(lambda x: reco_num_refer(x['Referred a Friend'], x['Number of Referrals']), axis=1)
test['Number of Referrals'] = test.apply(lambda x: reco_num_refer(x['Referred a Friend'], x['Number of Referrals']), axis=1)

#%%
# Tenure in Months = Total Charges / Monthly Charge 
# Total Charges, Monthly Charge -> Tenure in Months
def reco_tenure_by_normal_charge(tenure, ttl_charge, mon_charge):
    if np.isnan(tenure) == False:
        return tenure
    if np.isnan(ttl_charge) == False and np.isnan(mon_charge) == False and mon_charge != 0:
        return math.ceil(ttl_charge / mon_charge)
    return np.nan

train['Tenure in Months'] = train.apply(lambda x: reco_tenure_by_normal_charge(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)
test['Tenure in Months'] = test.apply(lambda x: reco_tenure_by_normal_charge(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)

# Tenure in Months, Monthly Charge -> Total Charges
def reco_ttl_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(ttl_charge) == False:
        return ttl_charge
    if np.isnan(tenure) == False and np.isnan(mon_charge) == False:
        return tenure * mon_charge
    return np.nan

train['Total Charges'] = train.apply(lambda x: reco_ttl_charge_by_tenure(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)
test['Total Charges'] = test.apply(lambda x: reco_ttl_charge_by_tenure(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)

# Tenure in Months, Total Charges -> Monthly Charge
def reco_mon_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(mon_charge) == False:
        return mon_charge
    if np.isnan(tenure) == False and np.isnan(ttl_charge) == False and tenure != 0:
        return ttl_charge / tenure
    return np.nan

train['Monthly Charge'] = train.apply(lambda x: reco_mon_charge_by_tenure(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)
test['Monthly Charge'] = test.apply(lambda x: reco_mon_charge_by_tenure(
    x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
)

#%%
# Tenure in Months = Total Long Distance Charges / Avg Monthly Long Distance Charges
# Total Long Distance Charges, Avg Monthly Long Distance Charges -> Tenure in Months
def reco_tenure_by_long_charge(tenure, ttl_charge, mon_charge):
    if np.isnan(tenure) == False:
        return tenure
    if np.isnan(ttl_charge) == False and np.isnan(mon_charge) == False and mon_charge != 0:
        return math.ceil(ttl_charge / mon_charge)
    return np.nan

train['Tenure in Months'] = train.apply(lambda x: reco_tenure_by_long_charge(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)
test['Tenure in Months'] = test.apply(lambda x: reco_tenure_by_long_charge(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)

# Tenure in Months, Avg Monthly Long Distance Charges -> Total Long Distance Charges
def reco_long_ttl_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(ttl_charge) == False:
        return ttl_charge
    if np.isnan(tenure) == False and np.isnan(mon_charge) == False:
        return tenure * mon_charge
    return np.nan

train['Total Long Distance Charges'] = train.apply(lambda x: reco_long_ttl_charge_by_tenure(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)
test['Total Long Distance Charges'] = test.apply(lambda x: reco_long_ttl_charge_by_tenure(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)

# Tenure in Months, Total Long Distance Charges -> Avg Monthly Long Distance Charges
def reco_long_mon_charge_by_tenure(tenure, ttl_charge, mon_charge):
    if np.isnan(mon_charge) == False:
        return mon_charge
    if np.isnan(tenure) == False and np.isnan(ttl_charge) == False and tenure != 0:
        return ttl_charge / tenure
    return np.nan

train['Avg Monthly Long Distance Charges'] = train.apply(lambda x: reco_long_mon_charge_by_tenure(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)
test['Avg Monthly Long Distance Charges'] = test.apply(lambda x: reco_long_mon_charge_by_tenure(
    x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
)

#%%
# Total Revenue = Total Charges - Total Refunds + Total Extra Data Charges + Total Long Distance Charges
# Total Charges, Total Refunds, Total Extra Data Charges, Total Long Distance Charges -> Total Revenue
def reco_revenue(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(revenue) == False:
        return revenue
    if np.isnan(ttl_charge) == False and np.isnan(refund) == False and np.isnan(data_charge) == False and np.isnan(long_charge) == False:
        return ttl_charge - refund + data_charge + long_charge
    return np.nan

train['Total Revenue'] = train.apply(lambda x: reco_revenue(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)
test['Total Revenue'] = test.apply(lambda x: reco_revenue(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

# Total Revenue, Total Refunds, Total Extra Data Charges, Total Long Distance Charges -> Total Charges
def reco_ttl_charge(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(ttl_charge) == False:
        return ttl_charge
    if np.isnan(revenue) == False and np.isnan(refund) == False and np.isnan(data_charge) == False and np.isnan(long_charge) == False:
        return revenue + refund - data_charge - long_charge
    return np.nan

train['Total Charges'] = train.apply(lambda x: reco_ttl_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)
test['Total Charges'] = test.apply(lambda x: reco_ttl_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

# Total Revenue, Total Charges, Total Extra Data Charges, Total Long Distance Charges -> Total Refunds
def reco_refund(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(refund) == False:
        return refund
    if np.isnan(revenue) == False and np.isnan(ttl_charge) == False and np.isnan(data_charge) == False and np.isnan(long_charge) == False:
        return ttl_charge + data_charge + long_charge - revenue
    return np.nan

train['Total Refunds'] = train.apply(lambda x: reco_refund(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)
test['Total Refunds'] = test.apply(lambda x: reco_refund(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

# Total Revenue, Total Charges, Total Refunds, Total Long Distance Charges -> Total Extra Data Charges
def reco_data_charge(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(data_charge) == False:
        return data_charge
    if np.isnan(revenue) == False and np.isnan(ttl_charge) == False and np.isnan(refund) == False and np.isnan(long_charge) == False:
        return revenue - ttl_charge + refund - long_charge
    return np.nan

train['Total Extra Data Charges'] = train.apply(lambda x: reco_data_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)
test['Total Extra Data Charges'] = test.apply(lambda x: reco_data_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

# Total Revenue, Total Charges, Total Refunds, Total Extra Data Charges -> Total Long Distance Charges
def reco_long_charge(revenue, ttl_charge, refund, data_charge, long_charge):
    if np.isnan(long_charge) == False:
        return long_charge
    if np.isnan(revenue) == False and np.isnan(ttl_charge) == False and np.isnan(refund) == False and np.isnan(data_charge) == False:
        return revenue - ttl_charge + refund - data_charge
    return np.nan

train['Total Long Distance Charges'] = train.apply(lambda x: reco_long_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)
test['Total Long Distance Charges'] = test.apply(lambda x: reco_long_charge(
    x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
)

#%%
# Tenure in Months = Total Charges / Monthly Charge 
# Tenure in Months = Total Long Distance Charges / Avg Monthly Long Distance Charges
# Total Revenue = Total Charges - Total Refunds + Total Extra Data Charges + Total Long Distance Charges

def cycle_update(df):
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
            nan_before += df[col].isnull().sum()

        df['Tenure in Months'] = df.apply(lambda x: reco_tenure_by_normal_charge(
            x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
        )
        df['Total Charges'] = df.apply(lambda x: reco_ttl_charge_by_tenure(
            x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
        )
        df['Monthly Charge'] = df.apply(lambda x: reco_mon_charge_by_tenure(
            x['Tenure in Months'], x['Total Charges'], x['Monthly Charge']), axis=1
        )
        df['Tenure in Months'] = df.apply(lambda x: reco_tenure_by_long_charge(
            x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
        )
        df['Total Long Distance Charges'] = df.apply(lambda x: reco_long_ttl_charge_by_tenure(
            x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
        )
        df['Avg Monthly Long Distance Charges'] = df.apply(lambda x: reco_long_mon_charge_by_tenure(
            x['Tenure in Months'], x['Total Long Distance Charges'], x['Avg Monthly Long Distance Charges']), axis=1
        )
        df['Total Revenue'] = df.apply(lambda x: reco_revenue(
            x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
        )
        df['Total Charges'] = df.apply(lambda x: reco_ttl_charge(
            x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
        )
        df['Total Refunds'] = df.apply(lambda x: reco_refund(
            x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
        )
        df['Total Extra Data Charges'] = df.apply(lambda x: reco_data_charge(
            x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
        )
        df['Total Long Distance Charges'] = df.apply(lambda x: reco_long_charge(
            x['Total Revenue'], x['Total Charges'], x['Total Refunds'], x['Total Extra Data Charges'], x['Total Long Distance Charges']), axis=1
        )

        iter += 1
        nan_after = 0
        for col in cycle:
            nan_after += df[col].isnull().sum()
        if nan_before == nan_after:
            break
    
    return df

train = cycle_update(train)
test = cycle_update(test)

#%%
# Multiple Lines -> Phone Service
def reco_phone(phone, multi):
    if pd.isnull(phone) == False:
        return phone
    if multi == 'Yes':
        return 'Yes'
    return np.nan

train['Phone Service'] = train.apply(lambda x: reco_phone(
    x['Phone Service'], x['Multiple Lines']), axis=1
)
test['Phone Service'] = test.apply(lambda x: reco_phone(
    x['Phone Service'], x['Multiple Lines']), axis=1
)

# Phone Service -> Multiple Lines
def reco_multi(phone, multi):
    if pd.isnull(multi) == False:
        return multi
    if phone == 'No':
        return 'No'
    return np.nan

train['Multiple Lines'] = train.apply(lambda x: reco_multi(
    x['Phone Service'], x['Multiple Lines']), axis=1
)
test['Multiple Lines'] = test.apply(lambda x: reco_multi(
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

train['Internet Service'] = train.apply(lambda x: reco_internet_from_internet_type(
    x['Internet Service'], x['Internet Type']), axis=1
)
test['Internet Service'] = test.apply(lambda x: reco_internet_from_internet_type(
    x['Internet Service'], x['Internet Type']), axis=1
)

# Avg Monthly GB Download -> Internet Service
def reco_internet_from_internet_type(internet, gb):
    if pd.isnull(internet) == False:
        return internet
    if gb > 0:
        return 'Yes'
    return np.nan

train['Internet Service'] = train.apply(lambda x: reco_internet_from_internet_type(
    x['Internet Service'], x['Avg Monthly GB Download']), axis=1
)
test['Internet Service'] = test.apply(lambda x: reco_internet_from_internet_type(
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
    train['Internet Service'] = train.apply(lambda x: reco_internet(
        x['Internet Service'], x[col]), axis=1
    )
    test['Internet Service'] = test.apply(lambda x: reco_internet(
        x['Internet Service'], x[col]), axis=1
    )

# Internet Service -> Internet Type
def reco_internet_type(internet, internet_type):
    if pd.isnull(internet_type) == False:
        return internet_type
    if internet == 'No':
        return 'None'
    return np.nan

train['Internet Type'] = train.apply(lambda x: reco_internet_type(
    x['Internet Service'], x['Internet Type']), axis=1
)
test['Internet Type'] = test.apply(lambda x: reco_internet_type(
    x['Internet Service'], x['Internet Type']), axis=1
)

# Internet Service -> Avg Monthly GB Download
def reco_GB(internet, gb):
    if pd.isnull(gb) == False:
        return gb
    if internet == 'No':
        return 0
    return np.nan

train['Avg Monthly GB Download'] = train.apply(lambda x: reco_GB(
    x['Internet Service'], x['Avg Monthly GB Download']), axis=1
)
test['Avg Monthly GB Download'] = test.apply(lambda x: reco_GB(
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
    train['Internet Service'] = train.apply(lambda x: reco_from_internet(
        x['Internet Service'], x[col]), axis=1
    )
    test['Internet Service'] = test.apply(lambda x: reco_from_internet(
        x['Internet Service'], x[col]), axis=1
    )

#%%
# drop redundant columns
# TBD (keep 30 65?)
redundant_cols = [
    'Under 30',
    'Senior Citizen',
    'Dependents',
    'Country',
    'State',
    'City',
    'Zip Code',
    'Lat Long',
    'Referred a Friend',
]

train_reco = train.drop(redundant_cols, axis=1)
test_reco = test.drop(redundant_cols, axis=1)

#%%
# categorize features
# TBD (real part?)
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

real_cols_fill_avg_int = [
    'Age',
    'Tenure in Months',
    'Avg Monthly GB Download',
    'Population',
    'Satisfaction Score',
]

real_cols_fill_avg_float = [
    'Avg Monthly Long Distance Charges',
    'Monthly Charge',
    'Total Charges',
    'Total Refunds',
    'Total Extra Data Charges',
    'Total Long Distance Charges',
    'Total Revenue',
]

real_cols_fill_zero = [
    
]

real_cols_fill_mode = [
    'Number of Dependents',
    'Number of Referrals',
]

other_cols_fill_mode = [
    'Latitude',
    'Longitude',
]

id_col = 'Customer ID'
label_col = 'Churn Category'
predictors = [x for x in train_reco.columns if x not in [label_col, id_col]]

#%%
# visualization
# for col in real_cols_fill_avg:
#     plot = data[[col]].plot(kind='hist', bins=30)
#     fig = plot.get_figure()
#     fig.savefig("./img/before_normalize/bn_" + col + ".png")

# for col in real_cols_fill_zero:
#     plot = data[[col]].plot(kind='hist', bins=30)
#     fig = plot.get_figure()
#     fig.savefig("./img/before_normalize/bn_" + col + ".png")

#%%
# fill real-valued nan by train data
for col in real_cols_fill_avg_int:
    train_reco[col].fillna(round(train_reco[col].mean()), inplace=True)
    test_reco[col].fillna(round(train_reco[col].mean()), inplace=True)

for col in real_cols_fill_avg_float:
    train_reco[col].fillna(train_reco[col].mean(), inplace=True)
    test_reco[col].fillna(train_reco[col].mean(), inplace=True)

for col in real_cols_fill_zero:
    train_reco[col].fillna(0, inplace=True)
    test_reco[col].fillna(0, inplace=True)

for col in real_cols_fill_mode:
    train_reco[col].fillna(train_reco[col].mode()[0], inplace=True)
    test_reco[col].fillna(train_reco[col].mode()[0], inplace=True)

for col in other_cols_fill_mode:
    train_reco[col].fillna(train_reco[col].mode()[0], inplace=True)
    test_reco[col].fillna(train_reco[col].mode()[0], inplace=True)

#%%
# fill categorical-valued nan by train data
# TBD (mode?)

#%%
# one hot
train_filled = pd.get_dummies(data=train_reco, columns=cat_cols, dummy_na=True)
test_filled = pd.get_dummies(data=test_reco, columns=cat_cols, dummy_na=True)

#%%
list_nan(train_filled)

#%%
# standard normalization
def std_normalize(s, mean_value, std_value):
    return (s - mean_value) / std_value

# min-max normalization
def min_max_normalize(s, min_value, max_value):
    return (s - min_value) / (max_value - min_value)

#%%
# normalization
if APPLY_NORMALIZE:
    if STD_NORMALIZE:
        for col in real_cols_fill_avg_int + real_cols_fill_avg_float + real_cols_fill_zero + real_cols_fill_mode:
            mean_value = train_filled[col].mean()
            std_value = train_filled[col].std()
            train_filled[col] = train_filled.apply(lambda x: std_normalize(x[col], mean_value, std_value), axis=1)
            test_filled[col] = test_filled.apply(lambda x: std_normalize(x[col], mean_value, std_value), axis=1)
    else:
        for col in real_cols_fill_avg_int + real_cols_fill_avg_float + real_cols_fill_zero + real_cols_fill_mode:
            min_value = train_filled[col].min()
            max_value = train_filled[col].max()
            train_filled[col] = train_filled.apply(lambda x: min_max_normalize(x[col], min_value, max_value), axis=1)
            test_filled[col] = test_filled.apply(lambda x: min_max_normalize(x[col], min_value, max_value), axis=1)

#%%
# map target label to numeric category
label_cat = dict({
    'No Churn' : 0,
    'Competitor' : 1,
    'Dissatisfaction' : 2,
    'Attitude' : 3,
    'Price' : 4,
    'Other' : 5,
    np.nan : -1
})
train_filled[label_col] = train_filled[label_col].map(label_cat)

#%%
###################
# Pre-processing End
###################

#%%
# prepare output
train_clean = train_filled[
    [ col for col in train_filled.columns if col != label_col ] + [label_col]
]
for col in train_clean.columns:
    nan_num = train_clean[col].isnull().sum()
    if nan_num > 0:
        print("{}: {}" .format(col, nan_num))

test_clean = test_filled[
    [ col for col in test_filled.columns if col != label_col ] + [label_col]
]
for col in test_clean.columns:
    nan_num = test_clean[col].isnull().sum()
    if nan_num > 0:
        print("{}: {}" .format(col, nan_num))

train_clean

#%%
# dump data
if APPLY_NORMALIZE:
    if STD_NORMALIZE:
        method = 'std'
    else:
        method = 'minmax'
    train_clean.to_csv('./preprocessed_data/train_data_' + method + '_normalized.csv', index=False)
    test_clean.to_csv('./preprocessed_data/test_data_' + method + '_normalized.csv', index=False)
else:
    train_clean.to_csv('./preprocessed_data/train_data.csv', index=False)
    test_clean.to_csv('./preprocessed_data/test_data.csv', index=False)