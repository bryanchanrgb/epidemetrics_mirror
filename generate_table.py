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
INTITALISE SCRIPT PARAMETERS
'''

SAVE_PLOTS=False
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
raw_mobility = raw_mobility[['countrycode','country','date','transit_stations','residential',
                             'workplace','parks','retail_recreation','grocery_pharmacy']]
raw_mobility = raw_mobility.sort_values(by=['countrycode','date']).reset_index(drop=True)
### Check no conflicting values for each country and date
assert not raw_mobility[['countrycode','country','date']].duplicated().any()

# GET GOVERNMENT RESPONSE TABLE
flags=['stringency_index','c1_school_closing','c2_workplace_closing','c3_cancel_public_events',
       'c4_restrictions_on_gatherings','c5_close_public_transport','c6_stay_at_home_requirements',
       'c7_restrictions_on_internal_movement','c8_international_travel_controls']

sql_command = """SELECT * FROM government_response"""
raw_government_response = pd.read_sql(sql_command, conn)
raw_government_response = raw_government_response.sort_values(by=['countrycode','date']).reset_index(drop=True)
raw_government_response = raw_government_response[['countrycode','country','date']+flags]
### Check no conflicting values for each country and date
assert not raw_government_response[['countrycode','country','date']].duplicated().any()


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

##EPIDEMIOLOGY PROCESSING LOOP
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode','country','date','confirmed','new_per_day'])
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
    epidemiology = pd.concat((epidemiology,data)).reset_index(drop=True)
    continue

##MOBILITY PROCESSING LOOP
countries = raw_mobility['countrycode'].unique()
mobility = pd.DataFrame(columns=['countrycode','country','date','transit_stations','residential',
                                     'workplace','parks','retail_recreation','grocery_pharmacy'])

for country in countries:
    data = raw_mobility[raw_mobility['countrycode']==country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0],data.index.values[-1])])
    data[['countrycode','country']] = data[['countrycode','country']].fillna(method='backfill')
    data[mobilities] = data[mobilities].interpolate(method='linear')
    data.reset_index(inplace=True)
    mobility = pd.concat((mobility,data)).reset_index(drop=True)
    continue

##GOVERNMENT_RESPONSE LOOP
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

'''
INITIALISE COLUMNS FOR MASTER TABLE
'''

##EPIDEMIOLOGY TABLE
epidemiology_columns={
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'date':np.empty(0),
    'confirmed':np.empty(0),
    'new_per_day':np.empty(0),
    'new_per_day_ma':np.empty(0),
    'new_per_day_smooth':np.empty(0),
    'peak_dates':np.empty(0),
    'peak_heights':np.empty(0),
    'peak_widths':np.empty(0),
    'peak_prominence':np.empty(0),
    'peak_width_boundaries':np.empty(0), #NOT INCLUDED IN FINAL TABLE
    'peak_width_heights':np.empty(0), #NOT INCLUDED IN FINAL TABLE
    'threshold_dates':np.empty(0),
    'threshold_average_height':np.empty(0),
    'threshold_max_height':np.empty(0)
    }

##MOBILITY TABLE
mobility_columns={
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'date':np.empty(0)
    }

for mobility_type in mobilities:
    mobility_columns[mobility_type+'_smooth'] = np.empty(0)
    mobility_columns[mobility_type] = np.empty(0)
    mobility_columns[mobility_type+'_peak_dates'] = np.empty(0)
    mobility_columns[mobility_type+'_trough_dates'] = np.empty(0)
    mobility_columns[mobility_type+'_peak_heights'] = np.empty(0)
    mobility_columns[mobility_type+'_trough_heights'] = np.empty(0)
    mobility_columns[mobility_type+'_peak_prominences'] = np.empty(0)
    mobility_columns[mobility_type+'_trough_prominences'] = np.empty(0)
    mobility_columns[mobility_type+'_peak_widths'] = np.empty(0)
    mobility_columns[mobility_type+'_trough_widths'] = np.empty(0)
    if SAVE_PLOTS:
        os.makedirs(PATH+'mobility/'+mobility_type+'/',exist_ok=True)

##GOVERNMENT_RESPONSE
government_response_columns={
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0)
}

'''
EPIDEMIOLOGY PROCESSING
'''

countries = np.sort(epidemiology['countrycode'].unique())

for country in tqdm(countries,desc='Processing Epidemiological Data'):
    data = epidemiology[epidemiology['countrycode'] == country]
