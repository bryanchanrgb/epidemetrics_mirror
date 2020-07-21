import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import os
import warnings
from tqdm import tqdm

from csaps import csaps
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import peak_prominences
warnings.filterwarnings('ignore')

'''
Initialise script parameters
'''

DISTANCE = 21  # Distance between peaks
THRESHOLD = 25  # Threshold for duration calculation
SMOOTH = 0.001  # Smoothing parameter for the spline fit
MIN_PERIOD = 14  # Minimum number of days above threshold
ROLLING_WINDOW = 3  # Window size for Moving Average for epidemiological data
PATH = './charts/final_figures/' #Path to save master table and corresponding plots

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()

# GET EPIDEMIOLOGY TABLE
source = "WRD_ECDC"
exclude = ['Other continent', 'Asia', 'Europe', 'America', 'Africa', 'Oceania']

sql_command = """SELECT * FROM epidemiology WHERE source = %(source)s"""
raw_epidemiology = pd.read_sql(sql_command, conn, params={'source': source})
raw_epidemiology = raw_epidemiology[raw_epidemiology['adm_area_1'].isnull()].sort_values(by=['countrycode', 'date'])
raw_epidemiology = raw_epidemiology[~raw_epidemiology['country'].isin(exclude)].reset_index(drop=True)
raw_epidemiology = raw_epidemiology[['countrycode','country','date','confirmed']]
### Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode','country','date']].duplicated().any()

# GET MOBILITY TABLE
source='GOOGLE_MOBILITY'
mobilities=['transit_stations','residential','workplace','parks','retail_recreation','grocery_pharmacy']

sql_command = """SELECT * FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility[['countrycode','country','date','transit_stations','residential','workplace','parks','retail_recreation','grocery_pharmacy']]
raw_mobility = raw_mobility.sort_values(by=['countrycode','date']).reset_index(drop=True)
### Check no conflicting values for each country and date
assert not raw_mobility[['countrycode','country','date']].duplicated().any()

# GET GOVERNMENT RESPONSE TABLE
sql_command = """SELECT * FROM government_response"""
raw_government_response = pd.read_sql(sql_command, conn)
raw_government_response = raw_government_response.sort_values(by=['countrycode','date']).reset_index(drop=True)
### Check no conflicting values for each country and date
assert not raw_government_response[['countrycode','country','date']].duplicated().any()


'''
PRE-PROCESSING for the main tables ['epidemiology','mobility','government_response']
Steps taken:
1) Filling gaps in epidemiological data through interpolation √
2) Filling gaps in mobility data through interpolation 
3) Filling gaps in government_response data through backfill 
5) Removing countries that are not in all three tables
6) Computing new cases per day for epidemiological data √
7) Replace negative/invalid new_cases_per_day with previous valid value √
'''

countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode','country','date','confirmed','new_per_day'])

##EPIDEMIOLOGY PROCESSING LOOP
for country in countries:
    data=raw_epidemiology[raw_epidemiology['countrycode']==country].set_index('date')
    data=data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']]=data[['countrycode','country']].fillna(method='backfill')
    data['confirmed']=data['confirmed'].interpolate(method='linear')
    data['new_per_day']=data['confirmed'].diff()
    data.reset_index(inplace=True)
    data['new_per_day'].iloc[np.array(data[data['new_per_day']<0].index)]=\
        data['new_per_day'].iloc[np.array(epidemiology[epidemiology['new_per_day']<0].index)-1]
    data['new_per_day']=data['new_per_day'].fillna(method='ffill')
    epidemiology = pd.concat((epidemiology,data))
    continue

##MOBILITY PROCESSING LOOP
for country in countries:
    continue

"""exclude=[]
for country in countries:
    if len(epidemiology[epidemiology['countrycode']==country])==0 or \
            len(epidemiology[epidemiology['countrycode'] == country]) == 0
"""
