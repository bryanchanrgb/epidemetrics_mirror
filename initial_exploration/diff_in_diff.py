import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import os
import warnings
from tqdm import tqdm
import datetime
import re

from csaps import csaps
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from scipy.signal import peak_prominences

from scipy.ndimage.interpolation import shift
import statsmodels.api as sm
warnings.filterwarnings('ignore')

'''
INTITALISE SCRIPT PARAMETERS
'''

SAVE_PLOTS=False
DISTANCE = 21  # Distance between peaks
THRESHOLD = 25  # Threshold for duration calculation
SMOOTH = 0.001  # Smoothing parameter for the spline fit
MIN_PERIOD = 14  # Minimum number of days above threshold
ROLLING_WINDOW = 3  # Window size for Moving Average for epidemiological data
OPTIMAL_LAG_TIME = 14 # +/- this time for optimal lag calculation
PATH = './charts/table_figures/' # Path to save master table and corresponding plots
PROMINENCE_THRESHOLD = 5
T0_THRESHOLD = 5 # T0 is defined as the first day where te smoothed number of new cases per day >= threshold

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
raw_epidemiology = raw_epidemiology[['countrycode','country','date','confirmed','dead']]
### Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode','date']].duplicated().any()

# GET MOBILITY TABLE
source='GOOGLE_MOBILITY'
mobilities=['transit_stations','residential','workplace','parks','retail_recreation','grocery_pharmacy']

sql_command = """SELECT * FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility[['countrycode','country','date']+mobilities]
raw_mobility = raw_mobility.sort_values(by=['countrycode','date']).reset_index(drop=True)
### Check no conflicting values for each country and date
assert not raw_mobility[['countrycode','date']].duplicated().any()

# GET GOVERNMENT RESPONSE TABLE
flags=['stringency_index','c1_school_closing','c2_workplace_closing','c3_cancel_public_events',
       'c4_restrictions_on_gatherings','c5_close_public_transport','c6_stay_at_home_requirements',
       'c7_restrictions_on_internal_movement']

sql_command = """SELECT * FROM government_response"""
raw_government_response = pd.read_sql(sql_command, conn)
raw_government_response = raw_government_response.sort_values(by=['countrycode','date']).reset_index(drop=True)
raw_government_response = raw_government_response[['countrycode','country','date']+flags]
raw_government_response = raw_government_response.sort_values(by=['country','date'])\
    .drop_duplicates(subset=['countrycode','date'])#.dropna(subset=['stringency_index'])
raw_government_response = raw_government_response.sort_values(by=['countrycode','date'])
### Check no conflicting values for each country and date
assert not raw_government_response[['countrycode','date']].duplicated().any()

# GET ADMINISTRATIVE DIVISION TABLE
sql_command = """SELECT * FROM administrative_division WHERE adm_level=0"""
map_data = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')

# GET POPULATION FROM WORLD BANK TABLE
indicator_code = 'SP.POP.TOTL'
sql_command = """SELECT countrycode, value, year FROM world_bank WHERE adm_area_1 IS NULL AND indicator_code = %(indicator_code)s"""
WB_statistics = pd.read_sql(sql_command, conn, params={'indicator_code': indicator_code})
# Check only 1 latest year value for population
assert len(WB_statistics) == len(WB_statistics['countrycode'].unique())
WB_statistics = WB_statistics.sort_values(by=['countrycode','year'], ascending=[True,False]).reset_index(drop=True)



'''
PRE-PROCESSING for the main tables ['epidemiology','mobility','government_response']
Steps taken:
1) Filling gaps in epidemiological data through interpolation √
2) Filling gaps in mobility data through interpolation √
3) Filling gaps in government_response data through backfill √
5) Removing countries that are not in all three tables √
6) Computing new cases per day for epidemiological data √
7) Replace negative/invalid new_cases_per_day with previous valid value √
'''

##EPIDEMIOLOGY PRE-PROCESSING LOOP
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode','country','date','confirmed','new_per_day','dead'])
for country in countries:
    data = raw_epidemiology[raw_epidemiology['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data['confirmed'] = data['confirmed'].interpolate(method='linear')
    data['new_per_day'] = data['confirmed'].diff()
    data.reset_index(inplace=True)
    data['new_per_day'].iloc[np.array(data[data['new_per_day']<0].index)] = \
        data['new_per_day'].iloc[np.array(epidemiology[epidemiology['new_per_day']<0].index)-1]
    data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
    data['dead'] = data['dead'].interpolate(method='linear')
    epidemiology = pd.concat((epidemiology,data)).reset_index(drop=True)
    continue

##MOBILITY PRE-PROCESSING LOOP
countries = raw_mobility['countrycode'].unique()
mobility = pd.DataFrame(columns=['countrycode','country','date']+mobilities)

for country in countries:
    data = raw_mobility[raw_mobility['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data[mobilities] = data[mobilities].interpolate(method='linear')
    data.reset_index(inplace=True)
    mobility = pd.concat((mobility,data)).reset_index(drop=True)
    continue

##GOVERNMENT_RESPONSE PRE-PROCESSING LOOP
countries = raw_government_response['countrycode'].unique()
government_response = pd.DataFrame(columns=['countrycode','country','date']+flags)

for country in countries:
    data = raw_government_response[raw_government_response['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data[flags] = data[flags].fillna(method='ffill')
    data.reset_index(inplace=True)
    government_response = pd.concat((government_response,data)).reset_index(drop=True)
    continue

##EXCLUSION LOOP (DATA FOR EACH COUNTRY MUST BE AVAILABLE IN ALL THREE TABLES)
countries = np.union1d(np.union1d(epidemiology['countrycode'].unique(),mobility['countrycode'].unique()),
                       government_response['countrycode'].unique())
exclude = []
for country in countries:
    if len(epidemiology[epidemiology['countrycode'] == country]) == 0 \
            or len(mobility[mobility['countrycode'] == country]) == 0 or \
            len(government_response[government_response['countrycode'] == country]) == 0:
        exclude.append(country)

epidemiology = epidemiology[~epidemiology['countrycode'].isin(exclude)]
mobility = mobility[~mobility['countrycode'].isin(exclude)]
government_response = government_response[~government_response['countrycode'].isin(exclude)]

### Ensure all countries appear in the three tables
assert np.all(epidemiology['countrycode'].unique()==mobility['countrycode'].unique())
assert np.all(government_response['countrycode'].unique()==mobility['countrycode'].unique())

countries = epidemiology['countrycode'].unique()

for country in countries[0:1]:
    country = 'ESP'
    ep_data = epidemiology[epidemiology['countrycode']==country][['date','new_per_day']]
    gov_data = government_response[government_response['countrycode']==country][['date','c6_stay_at_home_requirements']]
    mob_data = mobility[mobility['countrycode']==country][['date','residential']]
