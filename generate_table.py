import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import os
from math import floor
import shutil
import warnings
from tqdm import tqdm
import datetime
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

'''
INTITALISE SCRIPT PARAMETERS
'''

SAVE_PLOTS = False
SAVE_CSV = True
PLOT_PATH = './plots/'
CSV_PATH = './data/'
SMOOTH = 0.001
T_SEP = 21  # Minimum number of days apart between peaks for sub-algorithm A
V_SEP = 0  # Minimum vertical distance for sub-algorithm B
DISTANCE = 21       # Number of days apart 2 peaks must at least be for both to be considered genuine
ABS_PROMINENCE_THRESHOLD =  55      # Minumum prominence threshold (in absolute number of new cases)
ABS_PROMINENCE_THRESHOLD_UPPER =  500      # Prominence threshold (in absolute number of new cases)
POP_PROMINENCE_THRESHOLD = 0.05      # Prominence threshold (in number of new cases per 10k population)
RELATIVE_PROMINENCE_THRESHOLD = 0.7 # Prominence threshold for each peak, relative to its own magnitude
ABS_PROMINENCE_THRESHOLD_DEAD =  5      # Minimum prominence threshold (in absolute number of deaths per day)
POP_PROMINENCE_THRESHOLD_DEAD = 0.0015 # Prominence threshold (in number of deaths per day per 10k population)
ABS_PROMINENCE_THRESHOLD_UPPER_DEAD = 50    # Prominence threshold (in absolute number of deaths per day)
CLASS_1_THRESHOLD = 55             # Threshold in number of new cases per day (smoothed) to be considered entering first wave
CLASS_1_THRESHOLD_DEAD = 5          # Threshold in number of dead per day (smoothed) to be considered entering first wave for deaths
# Currently T0 is defined as the date of the 10th death
TEST_LAG = 0 # Lag between test date and test results
DEATH_LAG = 21 # Lag between confirmed and death. Ideally would be sampled from a random distribution of some sorts
SI_THRESHOLD = 60
MAX_CLASS = 10 # Any country greater than this class (more waves identified) will be grouped into Others instead
CUTOFF_DATE = datetime.date(2021,4,1)
d_match = 35
TESTS_THRESHOLD = 10

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()

# GET RAW EPIDEMIOLOGY TABLE
source = "WRD_WHO"
exclude = ['Other continent', 'Asia', 'Europe', 'America', 'Africa', 'Oceania','World']
cols = 'countrycode, country, date, confirmed, dead'
sql_command = """SELECT """ + cols + """ FROM epidemiology WHERE adm_area_1 IS NULL AND source = %(source)s"""
raw_epidemiology = pd.read_sql(sql_command, conn, params={'source':source})
raw_epidemiology = raw_epidemiology.sort_values(by=['countrycode', 'date'])
raw_epidemiology = raw_epidemiology[~raw_epidemiology['country'].isin(exclude)]
raw_epidemiology = raw_epidemiology[raw_epidemiology['date']<=CUTOFF_DATE].reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode', 'date']].duplicated().any()

# GET RAW GOVERNMENT RESPONSE TABLE
flags = ['c1_school_closing','c2_workplace_closing','c3_cancel_public_events','c4_restrictions_on_gatherings','c5_close_public_transport',
         'c6_stay_at_home_requirements','c7_restrictions_on_internal_movement','c8_international_travel_controls',
         'h2_testing_policy','h3_contact_tracing']
flag_thresholds = {'c1_school_closing': 3,
                   'c2_workplace_closing': 3,
                   'c3_cancel_public_events': 2,
                   'c4_restrictions_on_gatherings': 4,
                   'c5_close_public_transport': 2,
                   'c6_stay_at_home_requirements': 2,
                   'c7_restrictions_on_internal_movement': 2,
                   'c8_international_travel_controls': 4,
                   'h2_testing_policy': 3,
                   'h3_contact_tracing': 2}

cols = 'countrycode, country, date, stringency_index, ' +  ', '.join(flags)
sql_command = """SELECT """ + cols + """ FROM government_response"""
raw_government_response = pd.read_sql(sql_command, conn)
raw_government_response = raw_government_response.sort_values(by=['countrycode', 'date']).reset_index(drop=True)
raw_government_response = raw_government_response[['countrycode', 'country', 'date', 'stringency_index'] + flags]
raw_government_response = raw_government_response.sort_values(by=['country', 'date']) \
    .drop_duplicates(subset=['countrycode', 'date'])  # .dropna(subset=['stringency_index'])
raw_government_response = raw_government_response.sort_values(by=['countrycode', 'date'])
raw_government_response = raw_government_response[raw_government_response['date']<=CUTOFF_DATE].reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_government_response[['countrycode', 'date']].duplicated().any()

# GET ADMINISTRATIVE DIVISION TABLE (For plotting)
sql_command = """SELECT * FROM administrative_division WHERE adm_level=0"""
map_data = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')

# GET COUNTRY STATISTICS (2011 - 2019 est.)
indicator_codes = ('SP.POP.TOTL', 'EN.POP.DNST', 'NY.GNP.PCAP.PP.KD','SM.POP.NETM')
indicator_codes_name = {
    'SP.POP.TOTL': 'value',
    'EN.POP.DNST': 'population_density',
    'NY.GNP.PCAP.PP.KD': 'gni_per_capita',
    'SM.POP.NETM': 'net_migration'}
sql_command = """SELECT countrycode, indicator_code, value FROM world_bank WHERE 
adm_area_1 IS NULL AND indicator_code IN %(indicator_code)s"""
raw_wb_statistics = pd.read_sql(sql_command, conn, params={'indicator_code': indicator_codes}).dropna()
assert not raw_wb_statistics[['countrycode','indicator_code']].duplicated().any()
raw_wb_statistics = raw_wb_statistics.sort_values(by=['countrycode'], ascending=[True]).reset_index(drop=True)
wb_statistics = pd.DataFrame()
countries = raw_wb_statistics['countrycode'].unique()
for country in countries:
    data = dict()
    data['countrycode'] = country
    for indicator in indicator_codes:
        if len(raw_wb_statistics[(raw_wb_statistics['countrycode'] == country) &
                                 (raw_wb_statistics['indicator_code'] == indicator)]['value']) == 0:
            continue
        data[indicator_codes_name[indicator]] = raw_wb_statistics[
            (raw_wb_statistics['countrycode'] == country) &
            (raw_wb_statistics['indicator_code'] == indicator)]['value'].iloc[0]
    wb_statistics = wb_statistics.append(data,ignore_index=True)
wb_statistics['net_migration'] = wb_statistics['net_migration'].abs()

# TESTING DATA FROM OUR WORLD IN DATA
raw_testing_data = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/' +
                               'owid-covid-data.csv'
                               , parse_dates=['date'])[[
                                'iso_code', 'date', 'total_tests', 'new_tests', 'new_tests_smoothed', 'positive_rate']].rename(columns={'iso_code':'countrycode'})
raw_testing_data['date'] = raw_testing_data['date'].apply(lambda x:x.date())
raw_testing_data = raw_testing_data[raw_testing_data['date']<=CUTOFF_DATE].reset_index(drop=True)
                                                       
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 0 - PRE-PROCESSING
'''

# EPIDEMIOLOGY PROCESSING
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode', 'country', 'date', 'confirmed', 'new_per_day','dead_per_day'])
for country in tqdm(countries, desc='Pre-processing Epidemiological Data'):
    data = raw_epidemiology[raw_epidemiology['countrycode'] == country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
    data[['countrycode', 'country']] = data[['countrycode', 'country']].fillna(method='backfill')
    data['confirmed'] = data['confirmed'].interpolate(method='linear')
    data['new_per_day'] = data['confirmed'].diff()
    data.reset_index(inplace=True)
    data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index)] = \
        data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index) - 1]
    data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
    data['dead'] = data['dead'].interpolate(method='linear')
    data['dead_per_day'] = data['dead'].diff()
    data.reset_index(inplace=True)
    data['dead_per_day'].iloc[np.array(data[data['dead_per_day'] < 0].index)] = \
        data['dead_per_day'].iloc[np.array(data[data['dead_per_day'] < 0].index) - 1]
    data['dead_per_day'] = data['dead_per_day'].fillna(method='bfill')
    epidemiology = pd.concat((epidemiology, data)).reset_index(drop=True)
    continue

# GOVERNMENT_RESPONSE PROCESSING
countries = raw_government_response['countrycode'].unique()
government_response = pd.DataFrame(columns=['countrycode', 'country', 'date'] + flags)

for country in tqdm(countries, desc='Pre-processing Government Response Data'):
    data = raw_government_response[raw_government_response['countrycode'] == country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
    data[['countrycode', 'country']] = data[['countrycode', 'country']].fillna(method='backfill')
    data[flags] = data[flags].fillna(method='ffill')
    data.reset_index(inplace=True)
    government_response = pd.concat((government_response, data)).reset_index(drop=True)
    continue

# TESTING PROCESSING
countries = [country for country in raw_testing_data['countrycode'].unique()
             if not(pd.isnull(country)) and (len(country) == 3)] #remove some odd values in the countrycode column
testing = pd.DataFrame(columns=['countrycode','date', 'total_tests', 'new_tests', 'new_tests_smooth', 'positive_rate'])

for country in tqdm(countries, desc='Pre-processing Testing Data'):
    data = raw_testing_data[raw_testing_data['countrycode'] == country].reset_index(drop=True)
    if len(data['new_tests'].dropna()) == 0:
        continue
    data = data.iloc[
           data[(data['new_tests'].notnull())|(data['new_tests_smoothed'].notnull())].index[0]:data[(data['new_tests'].notnull())|(data['new_tests_smoothed'].notnull())].index[-1]].set_index('date')
    if len(data) > 0:
        data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
        data[['countrycode']] = data[['countrycode']].fillna(method='backfill')
        data[['total_tests']] = data[['total_tests']].interpolate(method='linear')
        data[['new_tests']] = data[['new_tests']].interpolate(method='linear')
        data.reset_index(inplace=True)
        testing = pd.concat((testing, data), ignore_index=True)
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 1 - PROCESSING TIME SERIES DATA
'''

# INITIALISE EPIDEMIOLOGY TIME SERIES
epidemiology_series = {
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0),
    'confirmed': np.empty(0),
    'new_per_day': np.empty(0),
    'new_per_day_smooth': np.empty(0),
    'dead': np.empty(0),
    'new_cases_per_10k': np.empty(0),
    'dead_per_day': np.empty(0),
    'dead_per_day_smooth': np.empty(0),
    'new_deaths_per_10k': np.empty(0),
    'tests': np.empty(0),
    'new_tests': np.empty(0),
    'new_tests_smooth': np.empty(0),
    'positive_rate': np.empty(0),
    'positive_rate_smooth': np.empty(0),
    'days_since_t0_10_dead':np.empty(0)
}

# INITIALISE GOVERNMENT_RESPONSE TIME SERIES
government_response_series = {
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0),
    'si': np.empty(0)
}

for flag in flags:
    government_response_series[flag] = np.empty(0)
    government_response_series[flag + '_days_above_threshold'] = np.empty(0)

if SAVE_PLOTS:
    os.makedirs(PLOT_PATH + 'government_response/', exist_ok=True)

'''
EPIDEMIOLOGY TIME SERIES PROCESSING
'''

countries = np.sort(epidemiology['countrycode'].unique())
for country in tqdm(countries, desc='Processing Epidemiological Time Series Data'):
    data = epidemiology[epidemiology['countrycode'] == country]
    testing_data = testing[testing['countrycode'] == country]

    tests = np.repeat(np.nan, len(data))
    new_tests = np.repeat(np.nan, len(data))
    new_tests_smooth = np.repeat(np.nan, len(data))
    positive_rate = np.repeat(np.nan, len(data))
    positive_rate_smooth = np.repeat(np.nan, len(data))

    #x = np.arange(len(data['date']))
    #y = data['new_per_day'].values
    #ys = csaps(x, y, x, smooth=SMOOTH)
    ys = data[['new_per_day','date']].rolling(window=7, on='date').mean()['new_per_day']
    #z = data['dead_per_day'].values
    #zs = csaps(x, z, x, smooth=SMOOTH)
    zs = data[['dead_per_day','date']].rolling(window=7, on='date').mean()['dead_per_day']

    if len(testing_data) > 1:
        tests = data[['date']].merge(
            testing_data[['date','total_tests']],how='left',on='date')['total_tests'].values

        if sum(~pd.isnull(testing_data['new_tests_smoothed'])) > 0: # if testing data has new_tests_smoothed, use this
            new_tests_smooth = data[['date']].merge(
                testing_data[['date','new_tests_smoothed']],how='left',on='date')['new_tests_smoothed'].values
        
        if sum(~pd.isnull(testing_data['new_tests'])) > 0:
            new_tests = data[['date']].merge(
                testing_data[['date','new_tests']],how='left',on='date')['new_tests'].values
        else:
            new_tests = new_tests_smooth
        
        if sum(~pd.isnull(testing_data['new_tests_smoothed'])) == 0 and \
            sum(~pd.isnull(testing_data['new_tests'])) > 0: # if there is no data in new_tests_smoothed, compute 7 day moving average
            new_tests_smooth = data[['date']].merge(
                testing_data[['date','new_tests']],how='left',on='date')[['new_tests','date']].rolling(window=7, on='date').mean()['new_tests']
                
        positive_rate[~np.isnan(new_tests)] = data['new_per_day'][~np.isnan(new_tests)] / new_tests[~np.isnan(new_tests)]
        positive_rate[positive_rate > 1] = np.nan
        positive_rate_smooth = np.array(pd.Series(positive_rate).rolling(window=7).mean())

        #x_testing = np.arange(len(new_tests[~np.isnan(new_tests)]))
        #y_testing = new_tests[~np.isnan(new_tests)]
        #ys_testing = csaps(x_testing, y_testing, x_testing, smooth=SMOOTH)

    population = np.nan if len(wb_statistics[wb_statistics['countrycode']==country]['value'])==0 else \
        wb_statistics[wb_statistics['countrycode']==country]['value'].iloc[0]
    t0_10_dead = np.nan if len(data[data['dead']>=10]['date']) == 0 else \
        data[data['dead']>=10]['date'].iloc[0]
    days_since_t0_10_dead = np.repeat(np.nan,len(data)) if pd.isnull(t0_10_dead) else \
        np.array([(date - t0_10_dead).days for date in data['date'].values])

    new_cases_per_10k = 10000 * (ys / population)
    new_deaths_per_10k = 10000 * (zs / population)

    epidemiology_series['countrycode'] = np.concatenate((
        epidemiology_series['countrycode'], data['countrycode'].values))
    epidemiology_series['country'] = np.concatenate(
        (epidemiology_series['country'], data['country'].values))
    epidemiology_series['date'] = np.concatenate(
        (epidemiology_series['date'], data['date'].values))
    epidemiology_series['confirmed'] = np.concatenate(
        (epidemiology_series['confirmed'], data['confirmed'].values))
    epidemiology_series['new_per_day'] = np.concatenate(
        (epidemiology_series['new_per_day'], data['new_per_day'].values))
    epidemiology_series['new_per_day_smooth'] = np.concatenate(
        (epidemiology_series['new_per_day_smooth'], ys))
    epidemiology_series['dead'] = np.concatenate(
        (epidemiology_series['dead'], data['dead'].values))
    epidemiology_series['dead_per_day'] = np.concatenate(
        (epidemiology_series['dead_per_day'], data['dead_per_day'].values))
    epidemiology_series['dead_per_day_smooth'] = np.concatenate(
        (epidemiology_series['dead_per_day_smooth'], zs))
    epidemiology_series['new_cases_per_10k'] = np.concatenate(
        (epidemiology_series['new_cases_per_10k'], new_cases_per_10k))
    epidemiology_series['new_deaths_per_10k'] = np.concatenate(
        (epidemiology_series['new_deaths_per_10k'], new_deaths_per_10k))
    epidemiology_series['tests'] = np.concatenate(
        (epidemiology_series['tests'], tests))
    epidemiology_series['new_tests'] = np.concatenate(
        (epidemiology_series['new_tests'], new_tests))
    epidemiology_series['new_tests_smooth'] = np.concatenate(
        (epidemiology_series['new_tests_smooth'], new_tests_smooth))
    epidemiology_series['positive_rate'] = np.concatenate(
        (epidemiology_series['positive_rate'], positive_rate))
    epidemiology_series['positive_rate_smooth'] = np.concatenate(
        (epidemiology_series['positive_rate_smooth'], positive_rate_smooth))
    epidemiology_series['days_since_t0_10_dead'] = np.concatenate(
        (epidemiology_series['days_since_t0_10_dead'], days_since_t0_10_dead))

'''
GOVERNMENT RESPONSE TIME SERIES PROCESSING
'''

countries = np.sort(government_response['countrycode'].unique())
for country in tqdm(countries, desc='Processing Government Response Time Series Data'):
    data = government_response[government_response['countrycode'] == country]

    government_response_series['countrycode'] = np.concatenate((
        government_response_series['countrycode'], data['countrycode'].values))
    government_response_series['country'] = np.concatenate(
        (government_response_series['country'], data['country'].values))
    government_response_series['date'] = np.concatenate(
        (government_response_series['date'], data['date'].values))
    government_response_series['si'] = np.concatenate(
        (government_response_series['si'], data['stringency_index'].values))

    for flag in flags:
        days_above = (data[flag] >= flag_thresholds[flag]).astype(int).values

        government_response_series[flag] = np.concatenate(
            (government_response_series[flag], data[flag].values))
        government_response_series[flag + '_days_above_threshold'] = np.concatenate(
            (government_response_series[flag + '_days_above_threshold'], days_above))

epidemiology_series = pd.DataFrame.from_dict(epidemiology_series)
government_response_series = pd.DataFrame.from_dict(government_response_series)

# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 2 - PROCESSING PANEL DATA 
'''
epidemiology_panel = pd.DataFrame()
# There are some countries with poor data quality in the ECDC data, insufficient for a clear classification
# They are manually labelled from visual inspection, and are defined here.
exclude_countries = ['CMR','COG','GNQ','BWA','ESH']
# Some countries also have poor data quality, but the overall shape of the wave is still visible:
# KGZ, CHL, NIC, KAZ, ECU, CHN
countries = [c for c in epidemiology['countrycode'].unique() if c in wb_statistics['countrycode'].values and c not in exclude_countries]
for country in tqdm(countries, desc='Processing Epidemiological Panel Data'):
    data = dict()
    data['peak_1'] = np.nan
    data['peak_2'] = np.nan
    data['peak_3'] = np.nan
    data['date_peak_1'] = np.nan
    data['date_peak_2'] = np.nan
    data['date_peak_3'] = np.nan
    data['first_wave_start'] = np.nan
    data['first_wave_end'] = np.nan
    data['second_wave_start'] = np.nan
    data['second_wave_end'] = np.nan
    data['third_wave_start'] = np.nan
    data['third_wave_end'] = np.nan
    data['countrycode'] = country
    country_series = epidemetrics._get_series(country,'new_per_day_smooth') \
        .merge(epidemetrics._get_series(country,'confirmed'),on=['date'],how='left') \
        .merge(epidemetrics._get_series(country,'dead'),on=['date'],how='left') \
        .merge(epidemetrics._get_series(country,'tests'),on=['date'],how='left')
    
    if len(country_series) < DISTANCE: # If the time series does not have sufficient length, skip the country
        continue
    data['country'] = epidemiology.loc[epidemiology['countrycode'] == country,'country'].values[0]
    data['population'] = np.nan if len(wb_statistics[wb_statistics['countrycode'] == country]) == 0 else \
        wb_statistics[wb_statistics['countrycode'] == country]['value'].values[0]
    data['t0_10_dead'] = np.nan if len(country_series[country_series['dead']>=10]['date']) == 0 else \
        country_series[country_series['dead']>=10]['date'].iloc[0]
    data['last_confirmed'] = country_series['confirmed'].iloc[-1]
    data['last_dead'] = country_series['dead'].iloc[-1]
    data['testing_available'] = True if len(country_series['tests'].dropna()) > 0 else False
    if data['testing_available']:
        data['last_tests'] = country_series['tests'].dropna().iloc[-1]
    else:
        data['last_tests'] = np.nan
    
    result_class, result_peaks = epidemetrics._classify(country)
    genuine_peaks = [int(a) for a in result_peaks['location']]
    data['class'] = result_class

    if data['class'] == 1 and not pd.isnull(data['t0_10_dead']):
        data['first_wave_start'] = data['t0_10_dead']
        data['first_wave_end'] = max(country_series['date'])
        data['dead_first_wave'] = country_series.loc[country_series['date']==data['first_wave_end'],'dead'].values[0] \
                                  - country_series.loc[country_series['date']==data['first_wave_start'],'dead'].values[0]
        data['tests_first_wave'] = country_series.loc[country_series['date']==data['first_wave_end'],'tests'].values[0] \
                                  - country_series.loc[country_series['date']==data['first_wave_start'],'tests'].values[0]
    if data['class'] >= 2:
        data['peak_1'] = country_series['new_per_day_smooth'].values[genuine_peaks[0]]
        data['date_peak_1'] = country_series['date'].values[genuine_peaks[0]]

        bases=[int(a) for a in np.append(result_peaks['left_base'],result_peaks['right_base'])]
        # for first wave left base, take the min of: T0, or the closest base to the left
        try:
            data['first_wave_start'] = min(data['t0_10_dead'],country_series['date'].values[max([b for b in bases if b<genuine_peaks[0]])]) \
                                        if not pd.isnull(data['t0_10_dead']) \
                                        else country_series['date'].values[max([b for b in bases if b<genuine_peaks[0]])]
            # for first wave right base, take the closest base to the right
            data['first_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[0]])]
            data['dead_first_wave'] = country_series.loc[country_series['date']==data['first_wave_end'],'dead'].values[0] \
                                      - country_series.loc[country_series['date']==data['first_wave_start'],'dead'].values[0]
            data['tests_first_wave'] = country_series.loc[country_series['date']==data['first_wave_end'],'tests'].values[0] \
                                      - country_series.loc[country_series['date']==data['first_wave_start'],'tests'].values[0]
        except:
            print(country+' first wave skipped')
            data['first_wave_start']=np.nan
            data['first_wave_end']=np.nan
            data['dead_first_wave']=np.nan
            data['tests_first_wave']=np.nan
    
    if data['class'] >= 3:
        try:
            data['second_wave_start'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[0]])]
            data['second_wave_end'] = max(country_series['date'])
            data['dead_second_wave'] = country_series.loc[country_series['date']==data['second_wave_end'],'dead'].values[0] \
                                      - country_series.loc[country_series['date']==data['second_wave_start'],'dead'].values[0]
            data['tests_second_wave'] = country_series.loc[country_series['date']==data['second_wave_end'],'tests'].values[0] \
                            - country_series.loc[country_series['date']==data['second_wave_start'],'tests'].values[0]
        except:
            print(country+' second wave skipped')
            data['second_wave_start']=np.nan
            data['second_wave_end']=np.nan
            data['dead_second_wave']=np.nan
            data['tests_second_wave']=np.nan
    if data['class'] >= 4:
        data['peak_2'] = country_series['new_per_day_smooth'].values[genuine_peaks[1]]
        data['date_peak_2'] = country_series['date'].values[genuine_peaks[1]]
        try:
            data['second_wave_start'] = country_series['date'].values[max([b for b in bases if b<genuine_peaks[1]])]
            data['second_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[1]])]
            data['dead_second_wave'] = country_series.loc[country_series['date']==data['second_wave_end'],'dead'].values[0] \
                                      - country_series.loc[country_series['date']==data['second_wave_start'],'dead'].values[0]
            data['tests_second_wave'] = country_series.loc[country_series['date']==data['second_wave_end'],'tests'].values[0] \
                            - country_series.loc[country_series['date']==data['second_wave_start'],'tests'].values[0]
        except:
            print(country+' second wave skipped')
            data['second_wave_start']=np.nan
            data['second_wave_end']=np.nan
            data['dead_second_wave']=np.nan
            data['tests_second_wave']=np.nan

    if data['class'] >= 5:
        try:
            data['third_wave_start'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[1]])]
            data['third_wave_end'] = max(country_series['date'])
            data['dead_third_wave'] = country_series.loc[country_series['date']==data['third_wave_end'],'dead'].values[0] \
                                      - country_series.loc[country_series['date']==data['third_wave_start'],'dead'].values[0]
            data['tests_third_wave'] = country_series.loc[country_series['date']==data['third_wave_end'],'tests'].values[0] \
                            - country_series.loc[country_series['date']==data['third_wave_start'],'tests'].values[0]
        except:
            print(country+' third wave skipped')
            data['third_wave_start']=np.nan
            data['third_wave_end']=np.nan
            data['dead_third_wave']=np.nan
            data['tests_third_wave']=np.nan
    if data['class'] >= 6:
        data['peak_3'] = country_series['new_per_day_smooth'].values[genuine_peaks[2]]
        data['date_peak_3'] = country_series['date'].values[genuine_peaks[2]]
        try:
            data['third_wave_start'] = country_series['date'].values[max([b for b in bases if b<genuine_peaks[2]])]
            data['third_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[2]])]
            data['dead_third_wave'] = country_series.loc[country_series['date']==data['third_wave_end'],'dead'].values[0] \
                                      - country_series.loc[country_series['date']==data['third_wave_start'],'dead'].values[0]
            data['tests_third_wave'] = country_series.loc[country_series['date']==data['third_wave_end'],'tests'].values[0] \
                            - country_series.loc[country_series['date']==data['third_wave_start'],'tests'].values[0]
        except:
            print(country+' third wave skipped')
            data['third_wave_start']=np.nan
            data['third_wave_end']=np.nan
            data['dead_third_wave']=np.nan
            data['tests_third_wave']=np.nan

    epidemiology_panel = epidemiology_panel.append(data,ignore_index=True)


government_response_panel = pd.DataFrame(columns=['countrycode', 'country', 'max_si','date_max_si',
                                                  'si_days_to_max_si', 'si_at_t0','si_at_peak_1',
                                                  'si_days_to_threshold',
                                                  'si_days_above_threshold','si_days_above_threshold_first_wave',
                                                  'si_integral'] +
                                                 [flag + '_at_t0' for flag in flags] +
                                                 [flag + '_at_peak_1' for flag in flags] +
                                                 [flag + '_days_to_threshold' for flag in flags] +
                                                 [flag + '_days_above_threshold' for flag in flags] +
                                                 [flag + '_days_above_threshold_first_wave' for flag in flags] +
                                                 [flag + '_raised' for flag in flags] +
                                                 [flag + '_lowered' for flag in flags] +
                                                 [flag + '_raised_again' for flag in flags])

countries = government_response['countrycode'].unique()
for country in tqdm(countries,desc='Processing Gov Response Panel Data'):
    data = dict()
    country_series = government_response_series[government_response_series['countrycode'] == country]
    data['countrycode'] = country
    data['country'] = country_series['country'].iloc[0]
    if all(pd.isnull(country_series['si'])): # if no values for SI, skip to next country
        continue
    data['max_si'] = country_series['si'].max()
    data['date_max_si'] = country_series[country_series['si'] == data['max_si']]['date'].iloc[0]
    population = np.nan if len(wb_statistics[wb_statistics['countrycode'] == country]['value']) == 0 else \
        wb_statistics[wb_statistics['countrycode'] == country]['value'].iloc[0]
    t0 = np.nan if len(epidemiology_panel[epidemiology_panel['countrycode']==country]['t0_10_dead']) == 0 \
        else epidemiology_panel[epidemiology_panel['countrycode']==country]['t0_10_dead'].iloc[0]
    data['si_days_to_max_si'] = np.nan if pd.isnull(t0) else (data['date_max_si'] - t0).days
    data['si_days_above_threshold'] = sum(country_series['si']>=SI_THRESHOLD)
    data['si_integral'] = np.trapz(y=country_series['si'].dropna(), 
                                   x=[(a-country_series['date'].values[0]).days for a in country_series['date'][~np.isnan(country_series['si'])]])
    # Initialize columns as nan first for potential missing values
    data['si_days_above_threshold_first_wave'] = np.nan
    data['si_at_t0'] = np.nan
    data['si_at_peak_1'] = np.nan
    for flag in flags:
        data[flag + '_raised'] = np.nan
        data[flag + '_lowered'] = np.nan
        data[flag + '_raised_again'] = np.nan
        data[flag + '_at_t0'] = np.nan
        data[flag + '_at_peak_1'] = np.nan
        data[flag + '_days_above_threshold_first_wave'] = np.nan
    if country in epidemiology_panel['countrycode'].values:
        date_peak_1 = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'date_peak_1'].values[0]
        first_wave_start = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_start'].values[0]
        first_wave_end = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_end'].values[0]
        if not pd.isnull(t0) and t0 in country_series['date']:
            # SI value at T0
            data['si_at_t0'] = country_series.loc[country_series['date']==t0,'si'].values[0]
            # days taken to reach threshold
            data['si_days_to_threshold'] = (min(country_series.loc[country_series['si']>=SI_THRESHOLD,'date']) - t0).days \
                                            if sum(country_series['si']>=SI_THRESHOLD) > 0 else np.nan
            for flag in flags:
                 data[flag + '_at_t0'] = country_series.loc[country_series['date']==t0,flag].values[0]
                 data[flag + '_days_to_threshold'] = (min(country_series.loc[country_series[flag]>=flag_thresholds[flag],'date']) - t0).days \
                                                      if sum(country_series[flag]>=flag_thresholds[flag]) > 0 else np.nan
        if not (pd.isnull(date_peak_1) or pd.isnull(first_wave_start) or pd.isnull(first_wave_end)) \
        and date_peak_1 in country_series['date']:
            # SI value at peak date
            data['si_at_peak_1'] = country_series.loc[country_series['date']==date_peak_1,'si'].values[0]
            # number of days SI above the threshold during the first wave
            data['si_days_above_threshold_first_wave'] = sum((country_series['si']>=SI_THRESHOLD)&
                                                             (country_series['date']>=first_wave_start)&
                                                             (country_series['date']<=first_wave_end))
            for flag in flags:
                # flag value at peak date
                data[flag + '_at_peak_1'] = country_series.loc[country_series['date']==date_peak_1,flag].values[0]
                # number of days each flag above threshold during first wave
                data[flag + '_days_above_threshold_first_wave'] = country_series[
                    (country_series['date'] >= first_wave_start)&
                    (country_series['date'] <= first_wave_end)][flag + '_days_above_threshold'].sum()
    for flag in flags:
        days_above = pd.Series(country_series[flag + '_days_above_threshold'])
        waves = [[cat[1], grp.shape[0]] for cat, grp in
                 days_above.groupby([days_above.ne(days_above.shift()).cumsum(), days_above])]
        if len(waves) >= 2:
            data[flag + '_raised'] = country_series['date'].iloc[waves[0][1]]
        if len(waves) >= 3:
            data[flag + '_lowered'] = country_series['date'].iloc[
                waves[0][1] + waves[1][1]]
        if len(waves) >= 4:
            data[flag + '_raised_again'] = country_series['date'].iloc[
                waves[0][1] + waves[1][1] + waves[2][1]]
        data[flag + '_days_above_threshold'] = country_series[flag + '_days_above_threshold'].sum()

    government_response_panel = government_response_panel.append(data,ignore_index=True) 

#%% -------------------------------------------------------------------------------------------------------------------- #
'''
PART 3 - FIGURE 1
'''
'''
start_date = epidemiology_series['date'].min()
figure_1 = pd.DataFrame(columns=['countrycode', 'country', 'days_to_t0_10_dead', 'class',
                                 'new_cases_per_10k', 'new_deaths_per_10k','geometry'])
figure_1 = epidemiology_series[['countrycode', 'country','new_cases_per_10k', 'new_deaths_per_10k']]
figure_1 = figure_1.merge(epidemiology_panel[['countrycode', 't0_10_dead', 'class']], on=['countrycode'], how='left')
figure_1['days_to_t0_10_dead'] = (figure_1['t0_10_dead']-start_date).apply(lambda x: x.days)
figure_1 = figure_1.drop(columns=['t0_10_dead'])
figure_1 = figure_1.merge(map_data[['countrycode','geometry']], on=['countrycode'], how='left')

if SAVE_CSV:
    figure_1.to_csv(CSV_PATH + 'figure_1.csv', sep=';')
    figure_1.astype({'geometry': str}).to_csv(CSV_PATH + 'figure_1.csv', sep=';')
'''
countries=['ZMB','GBR','GHA','CRI']
figure_1_series = epidemiology_series.loc[epidemiology_series['countrycode'].isin(countries),
    ['country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
    'dead_per_day', 'dead_per_day_smooth']]

figure_1_panel = epidemiology_panel.loc[epidemiology_panel['countrycode'].isin(countries),
    ['class','country', 'countrycode','population','t0_10_dead',
     'date_peak_1','peak_1','first_wave_start','first_wave_end',
     'date_peak_2','peak_2','second_wave_start','second_wave_end',
     'date_peak_3','peak_3','third_wave_start','third_wave_end']]

if SAVE_CSV:
    figure_1_series.to_csv(CSV_PATH + 'figure_1_series.csv')
    figure_1_panel.to_csv(CSV_PATH + 'figure_1_panel.csv')
    
#%% -------------------------------------------------------------------------------------------------------------------- #
'''
PART 4 - FIGURE 2
'''
countries=['ITA','FRA','USA']
figure_2 = epidemiology_series.loc[epidemiology_series['countrycode'].isin(countries),
    ['country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
    'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smooth', 'positive_rate','positive_rate_smooth']]

if SAVE_CSV:
    figure_2.to_csv(CSV_PATH + 'figure_2.csv')
#%% -------------------------------------------------------------------------------------------------------------------- #
'''
PART 5 - FIGURE 3
'''

figure_3_wave_level = pd.DataFrame()
countries = epidemiology_panel['countrycode'].unique()
for country in tqdm(countries, desc='Processing figure 3 wave level data'):       
       data = dict()
       data['dead_during_wave'] = np.nan
       data['tests_during_wave'] = np.nan
       data['si_integral_during_wave'] = np.nan
       data['countrycode'] = country
       data['country'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'country'].values[0]
       data['class'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'class'].values[0]
       data['population'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'population'].values[0]
       if data['class'] >= 1:
            # First wave
            data['wave'] = 1
            data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_start'].values[0]
            data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_end'].values[0]
            data['t0_10_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_10_dead'].values[0]
            data['dead_during_wave'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'dead_first_wave'].values[0]
            data['tests_during_wave'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'tests_first_wave'].values[0]
            
            # if tests during first wave is na due to missing data, linear interpolate low test numbers
            if pd.isnull(data['tests_during_wave']):
                country_series = epidemiology_series[epidemiology_series['countrycode']==country]
                if not pd.isnull(data['wave_start']) and not np.all(pd.isnull(country_series['tests'])):
                    min_date = min(country_series['date'])
                    min_tests = np.nanmin(country_series['tests'])
                    if pd.isnull(country_series.loc[country_series['date']==min_date,'tests'].values[0]) \
                    and min_tests <= 1000:
                        country_series.loc[country_series['date']==min_date,'tests'] = 0 
                        country_series['tests'] = country_series['tests'].interpolate(method='linear')
                    if not pd.isnull(country_series.loc[country_series['date']==data['wave_start'],'tests'].values[0]) and not pd.isnull(country_series.loc[country_series['date']==data['wave_end'],'tests'].values[0]):
                        data['tests_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'tests'].values[0] - \
                                                    country_series.loc[country_series['date']==data['wave_start'],'tests'].values[0]
            
            si_series = government_response_series.loc[(government_response_series['countrycode'] == country) & 
                                                    (government_response_series['date'] >= data['wave_start']) &
                                                    (government_response_series['date'] <= data['wave_end']), ['si','date']]
            if len(si_series) == 0:
                data['si_integral_during_wave'] = np.nan
            else:
                data['si_integral_during_wave'] = np.trapz(y=si_series['si'].dropna(),x=[(a-si_series['date'].values[0]).days for a in si_series['date'][~np.isnan(si_series['si'])]])
            figure_3_wave_level = figure_3_wave_level.append(data, ignore_index=True)

            if data['class'] >= 3:
                # Second wave
                country_series = epidemiology_series[epidemiology_series['countrycode']==country]
                data['wave'] = 2
                data['t0_10_dead'] = np.nan
                data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'second_wave_start'].values[0]
                data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'second_wave_end'].values[0]
                data['dead_during_wave'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'dead_second_wave'].values[0]
                data['tests_during_wave'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'tests_second_wave'].values[0]
                dead_at_start = country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]
                data['t0_10_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                        (country_series['date']<=data['wave_end']) & \
                                                        (country_series['dead'] >= dead_at_start + 10), \
                                                        'date']
                if len(data['t0_10_dead']) > 0:
                    data['t0_10_dead'] = data['t0_10_dead'].values[0]
                else:
                    data['t0_10_dead'] = np.nan
                si_series = government_response_series.loc[(government_response_series['countrycode'] == country) & 
                                                        (government_response_series['date'] >= data['wave_start']) &
                                                        (government_response_series['date'] <= data['wave_end']), ['si','date']]
                if len(si_series) == 0:
                    data['si_integral_during_wave'] = np.nan
                else:
                    data['si_integral_during_wave'] = np.trapz(y=si_series['si'].dropna(),x=[(a-si_series['date'].values[0]).days for a in si_series['date'][~np.isnan(si_series['si'])]])
                figure_3_wave_level = figure_3_wave_level.append(data, ignore_index=True)

                if data['class'] >= 5:
                    # third wave
                    data['wave'] = 3
                    data['t0_10_dead'] = np.nan
                    data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'third_wave_start'].values[0]
                    data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'third_wave_end'].values[0]
                    data['dead_during_wave'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'dead_third_wave'].values[0]
                    data['tests_during_wave'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'tests_third_wave'].values[0]
                    dead_at_start = country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]
                    data['t0_10_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                            (country_series['date']<=data['wave_end']) & \
                                                            (country_series['dead'] >= dead_at_start + 10), \
                                                            'date']
                    if len(data['t0_10_dead']) > 0:
                        data['t0_10_dead'] = data['t0_10_dead'].values[0]
                    else:
                        data['t0_10_dead'] = np.nan
                    si_series = government_response_series.loc[(government_response_series['countrycode'] == country) & 
                                                            (government_response_series['date'] >= data['wave_start']) &
                                                            (government_response_series['date'] <= data['wave_end']), ['si','date']]
                    if len(si_series) == 0:
                        data['si_integral_during_wave'] = np.nan
                    else:
                        data['si_integral_during_wave'] = np.trapz(y=si_series['si'].dropna(),x=[(a-si_series['date'].values[0]).days for a in si_series['date'][~np.isnan(si_series['si'])]])
                    figure_3_wave_level = figure_3_wave_level.append(data, ignore_index=True)
if SAVE_CSV:
    figure_3_wave_level.to_csv(CSV_PATH + '/figure_3_wave_level.csv')

class_coarse = {
    0:'EPI_OTHER',
    1:'EPI_FIRST_WAVE',
    2:'EPI_FIRST_WAVE',
    3:'EPI_SECOND_WAVE',
    4:'EPI_SECOND_WAVE',
    5:'EPI_THIRD_WAVE',
    6:'EPI_THIRD_WAVE',
    7:'EPI_FOURTH_WAVE',
    8:'EPI_FOURTH_WAVE',
}

# figure_3_total: all waves
data = epidemiology_panel[['countrycode','country','class','last_confirmed','last_dead','last_tests',
                           't0_10_dead','population']].merge(
            government_response_panel[['countrycode','si_integral']],on='countrycode',how='left')
data['class_coarse'] = [class_coarse[x] if x in class_coarse.keys() else 'EPI_OTHER' for x in data['class'].values]
data['last_confirmed_per_10k'] = 10000 * epidemiology_panel['last_confirmed'] / epidemiology_panel['population']
data['last_dead_per_10k'] = 10000 * epidemiology_panel['last_dead'] / epidemiology_panel['population']
data['last_tests_per_10k'] = 10000 * epidemiology_panel['last_tests'] / epidemiology_panel['population']
data['first_date_si_above_threshold'] = np.nan
for flag in flags:
    data['first_date_'+flag[0:2]+'_above_threshold'] = np.nan
for country in tqdm(epidemiology_panel.countrycode):
    gov_country_series = government_response_series[government_response_series['countrycode']==country]
    country_series = epidemiology_series[epidemiology_series['countrycode']==country]
    if sum(gov_country_series['si']>=SI_THRESHOLD) > 0:
        data.loc[data['countrycode']==country,'first_date_si_above_threshold'] = min(gov_country_series.loc[gov_country_series['si']>=SI_THRESHOLD,'date'])
    for flag in flags:
        if sum(gov_country_series[flag]>=flag_thresholds[flag]) > 0:
            data.loc[data['countrycode']==country,'first_date_'+flag[0:2]+'_above_threshold'] = min(gov_country_series.loc[gov_country_series[flag]>=flag_thresholds[flag],'date'])
            if not pd.isnull(data.loc[data['countrycode']==country,'t0_10_dead']).values[0]:
                data.loc[data['countrycode']==country,flag[0:2]+'_response_time'] = (data.loc[data['countrycode']==country,'first_date_'+flag[0:2]+'_above_threshold'].values[0]-data.loc[data['countrycode']==country,'t0_10_dead'].values[0]).days
    tests_threshold_pop = TESTS_THRESHOLD * data.loc[data['countrycode']==country,'population'].values[0] / 10000 
    if sum(country_series['tests']>=tests_threshold_pop) > 0:
        data.loc[data['countrycode']==country,'first_date_tests_above_threshold'] = min(country_series.loc[country_series['tests']>=tests_threshold_pop,'date'])
        if not pd.isnull(data.loc[data['countrycode']==country,'t0_10_dead']).values[0]:
            data.loc[data['countrycode']==country,'testing_response_time'] = (data.loc[data['countrycode']==country,'first_date_tests_above_threshold'].values[0]-data.loc[data['countrycode']==country,'t0_10_dead'].values[0]).days
 
figure_3_total = data


figure_3_all = figure_3_wave_level.merge(
        figure_3_total[['countrycode','class_coarse','si_integral','last_dead_per_10k','last_tests_per_10k',
                        'first_date_si_above_threshold','first_date_c3_above_threshold','first_date_tests_above_threshold',
                        'c3_response_time','testing_response_time']]
                        ,on='countrycode',how='left')

if SAVE_CSV:
    figure_3_all.to_csv(CSV_PATH + 'figure_3_all.csv')

#%% -------------------------------------------------------------------------------------------------------------------- #
'''
PART 6 - FIGURE 4
'''
#Pulling Shape Files from OxCOVID (Indexed by GID)
sql_command = """SELECT * FROM administrative_division WHERE countrycode='USA'"""
usa_map = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')

#Pulling USA populations from JHU DB (Indexed by FIPS)
usa_populations = pd.read_csv('https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/' +
                              'UID_ISO_FIPS_LookUp_Table.csv')
#Pulling Case Data from NYT DB (Indexed by FIPS)
usa_cases = pd.read_csv('https://github.com/nytimes/covid-19-data/raw/master/us-counties.csv')

#Get only the peak dates
dates = ['2020-04-08','2020-07-21','2021-01-04']
usa_cases = usa_cases.loc[usa_cases['date'].isin(dates),]

#Using OxCOVID translation csv to map US county FIPS code to GIDs for map_data matching
translation_csv = pd.read_csv('https://github.com/covid19db/fetchers-python/raw/master/' +
                              'src/plugins/USA_NYT/translation.csv')

figure_4 = usa_cases.merge(translation_csv[['input_adm_area_1','input_adm_area_2','gid']],
    left_on=['state','county'], right_on=['input_adm_area_1','input_adm_area_2'], how='left').merge(
    usa_populations[['FIPS','Population']], left_on=['fips'], right_on=['FIPS'], how='left')

figure_4 = figure_4[['date', 'gid', 'fips', 'cases', 'Population']].sort_values(by=['gid','date']).dropna(subset=['gid'])
figure_4 = usa_map[['gid','geometry']].merge(figure_4, on=['gid'], how='right')

if SAVE_CSV:
    #figure_4.to_csv(CSV_PATH + 'figure_4.csv', sep=';')
    figure_4.astype({'geometry': str}).to_csv(CSV_PATH + 'figure_4.csv', sep=';')
#%% -------------------------------------------------------------------------------------------------------------------- #
  
'''
PART 6.5 - FIGURE 4A STACKED AREA PLOT BY STATE
'''    
# GET RAW CONFIRMED CASES TABLE FOR USA STATES
cols = 'countrycode, adm_area_1, date, confirmed'
sql_command = """SELECT """ + cols + """ FROM epidemiology WHERE countrycode = 'USA' AND source = 'USA_NYT' AND adm_area_1 IS NOT NULL AND adm_area_2 IS NULL"""
raw_usa = pd.read_sql(sql_command, conn)
raw_usa = raw_usa.sort_values(by=['adm_area_1', 'date']).reset_index(drop=True)
raw_usa = raw_usa[raw_usa['date']<=CUTOFF_DATE].reset_index(drop=True)

states = raw_usa['adm_area_1'].unique()
figure_4a = pd.DataFrame(columns=['countrycode', 'adm_area_1','date','confirmed','new_per_day','new_per_day_smooth'])
for state in tqdm(states, desc='Processing USA Epidemiological Data'):
    data = raw_usa[raw_usa['adm_area_1'] == state].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
    data['confirmed'] = data['confirmed'].interpolate(method='linear')
    data['new_per_day'] = data['confirmed'].diff()
    data.reset_index(inplace=True)
    data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index)] = \
        data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index) - 1]
    data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
    #x = np.arange(len(data['date']))
    #y = data['new_per_day'].values
    #ys = csaps(x, y, x, smooth=SMOOTH)
    ys = data[['new_per_day','date']].rolling(window=7, on='date').mean()['new_per_day']
    data['new_per_day_smooth'] = ys
    figure_4a = pd.concat((figure_4a, data)).reset_index(drop=True)
    continue

# Get latitude and longitude for states
sql_command = """SELECT adm_area_1, latitude, longitude FROM administrative_division WHERE adm_level=1 AND countrycode='USA'"""
states_lat_long = pd.read_sql(sql_command, conn)
figure_4a = figure_4a.merge(states_lat_long, on='adm_area_1')

if SAVE_CSV:
    figure_4a.to_csv(CSV_PATH + 'figure_4a.csv', sep=',')


#%% -------------------------------------------------------------------------------------------------------------------- #
'''
PART 8 - SAVING TABLE 1
'''

'''
Number of countries √
Duration of first wave of cases [days] (epi panel) √
Duration of second wave of cases [days] (epi panel) √
Average Days to t0 (epi panel) √
Average GDP per capita (wb panel) √
Population density (wb panel) √
Immigration (wb panel) √
Days with “Stay at home” (gov panel) √
Government response time [days between T0 and peak SI] (gov panel) √
Government response time [T0 to C6 flag raised] (gov panel) √
Number of new cases per day per 10,000 first peak (epi panel) √
Number of new cases per day per 10,000 second peak (epi panel) √
Number of deaths per day per 10,000 at first peak (epi panel) √
Number of deaths per day per 10,000 at second peak (epi panel) √
Case fatality rate first peak (epi panel) √
Case fatality rate second peak (epi panel) √
Peak date of New Cases (days since t0) (epi panel) √
Peak date of Stringency (days since t0) (gov panel) √
Peak date of Residential Mobility (days since t0) (mobility panel)  √
Quarantine Fatigue - Residential Mobility Correlation with time (First Wave Only) (mobility panel) √
'''

start_date = epidemiology_series['date'].min()
data = epidemiology_panel[
    ['countrycode', 'country', 'class', 't0_relative', 'peak_1_per_10k', 'peak_2_per_10k', 'peak_1_dead_per_10k',
     'peak_2_dead_per_10k', 'first_wave_start', 'first_wave_end', 'second_wave_start',
     'second_wave_end', 'date_peak_1', 'date_peak_2']].merge(
    wb_statistics[[
        'countrycode', 'gni_per_capita', 'net_migration', 'population_density']], on=['countrycode'], how='left').merge(
    government_response_panel[[
        'countrycode', 'date_max_si']], on=['countrycode'], how='left').merge(
    mobility_panel[[
        'countrycode', 'residential_max_date', 'residential_quarantine_fatigue']], on=['countrycode'], how='left')

table1 = pd.DataFrame()
table1['countrycode'] = data['countrycode']
table1['class'] = data['class']
table1['duration_first_wave'] = (data['first_wave_end'] - data['first_wave_start']).apply(lambda x: x.days)
table1['duration_second_wave'] = (data['second_wave_end'] - data['second_wave_start']).apply(lambda x: x.days)
table1['days_to_t0'] = (data['t0_relative'] - start_date).apply(lambda x: x.days)
table1['gni_per_capita'] = data['gni_per_capita']
table1['population_density'] = data['population_density']
table1['net_migration'] = data['net_migration']
table1['new_cases_per_day_peak_1_per_10k'] = data['peak_1_per_10k']
table1['new_cases_per_day_peak_2_per_10k'] = data['peak_2_per_10k']
table1['new_deaths_per_day_peak_1_per_10k'] = data['peak_1_dead_per_10k']
table1['new_deaths_per_day_peak_2_per_10k'] = data['peak_2_dead_per_10k']
table1['new_cases_per_day_peak_1_date'] = (data['date_peak_1'] - data['t0_relative']).apply(lambda x: x.days)
table1['new_cases_per_day_peak_2_date'] = (data['date_peak_2'] - data['t0_relative']).apply(lambda x: x.days)
table1['government_response_peak_date'] = (data['date_max_si'] - data['t0_relative']).apply(lambda x: x.days)
table1['mobility_residential_peak_date'] = (data['residential_max_date'] - data['t0_relative']).apply(lambda x: x.days)
table1['mobility_quarantine_fatigue'] = data['residential_quarantine_fatigue']
table1 = table1[(table1['class']!=0) & (~table1['days_to_t0'].isna())]
table1 = table1[['countrycode','class']].groupby(by='class').count().join(table1.groupby(by=['class']).mean())
table1 = table1.transpose()

if SAVE_CSV:
    table1.to_csv(CSV_PATH + 'table1.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
SAVING TIMESTAMP
'''

if SAVE_CSV:
    np.savetxt(CSV_PATH + 'last_updated.txt', 
               ['Update date: ' + datetime.datetime.today().date().strftime('%Y-%m-%d') + \
               '\nCutoff date: ' + CUTOFF_DATE.strftime('%Y-%m-%d')],
               fmt='%s')
# -------------------------------------------------------------------------------------------------------------------- #
