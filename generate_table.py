import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import os
import shutil
import warnings
from tqdm import tqdm
import datetime
from skimage.transform import resize
#from csaps import csaps
from scipy.signal import find_peaks
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

'''
INTITALISE SCRIPT PARAMETERS
'''

SAVE_PLOTS = True
SAVE_CSV = True
PLOT_PATH = './plots/'
CSV_PATH = './data/'
SMOOTH = 0.001
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
# Currently T0 is defined as the date of the 1st death (alternatively t0_5_dead or t0_10_dead for the 5th and 10th death)
ABSOLUTE_T0_THRESHOLD = 1000    # Define an alternative T0 as the date reaching this threshold in total cases
POP_RELATIVE_T0_THRESHOLD = 0.05 # T0 threshold in cumulative cases per 10k population
TEST_LAG = 0 # Lag between test date and test results
DEATH_LAG = 21 # Lag between confirmed and death. Ideally would be sampled from a random distribution of some sorts
SI_THRESHOLD = 60
CUTOFF_DATE = datetime.date(2020,11,30)

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()

# GET RAW EPIDEMIOLOGY TABLE
source = "WRD_ECDC"
exclude = ['Other continent', 'Asia', 'Europe', 'America', 'Africa', 'Oceania','World']
cols = 'countrycode, country, date, confirmed, dead'
sql_command = """SELECT """ + cols + """ FROM epidemiology WHERE adm_area_1 IS NULL AND source = %(source)s"""
raw_epidemiology = pd.read_sql(sql_command, conn, params={'source':source})
raw_epidemiology = raw_epidemiology.sort_values(by=['countrycode', 'date'])
raw_epidemiology = raw_epidemiology[~raw_epidemiology['country'].isin(exclude)]
raw_epidemiology = raw_epidemiology[raw_epidemiology['date']<=CUTOFF_DATE].reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode', 'date']].duplicated().any()

# GET RAW MOBILITY TABLE
source = 'GOOGLE_MOBILITY'
mobilities = ['residential','workplace','transit_stations','retail_recreation']
cols = 'countrycode, country, date, ' +  ', '.join(mobilities)
sql_command = """SELECT """ + cols + """ FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility.sort_values(by=['countrycode', 'date']).reset_index(drop=True)
raw_mobility = raw_mobility[raw_mobility['date']<=CUTOFF_DATE].reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_mobility[['countrycode', 'date']].duplicated().any()

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

# MOBILITY PROCESSING
countries = raw_mobility['countrycode'].unique()
mobility = pd.DataFrame(columns=['countrycode', 'country', 'date'] + mobilities)

for country in tqdm(countries, desc='Pre-processing Mobility Data'):
    data = raw_mobility[raw_mobility['countrycode'] == country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
    data[['countrycode', 'country']] = data[['countrycode', 'country']].fillna(method='backfill')
    data[mobilities] = data[mobilities].interpolate(method='linear')
    data.reset_index(inplace=True)
    mobility = pd.concat((mobility, data)).reset_index(drop=True)
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
    'days_since_t0': np.empty(0),
    'new_cases_per_10k': np.empty(0),
    'dead_per_day': np.empty(0),
    'dead_per_day_smooth': np.empty(0),
    'new_deaths_per_10k': np.empty(0),
    'tests': np.empty(0),
    'new_tests': np.empty(0),
    'new_tests_smooth': np.empty(0),
    'positive_rate': np.empty(0),
    'positive_rate_smooth': np.empty(0),
    'days_since_t0_pop':np.empty(0),
    'days_since_t0_1_dead':np.empty(0),
    'days_since_t0_5_dead':np.empty(0),
    'days_since_t0_10_dead':np.empty(0)
}

# INITIALISE MOBILITY TIME SERIES
mobility_series = {
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0)
}

if SAVE_PLOTS:
    os.makedirs(PLOT_PATH + 'mobility/', exist_ok=True)

for mobility_type in mobilities:
    mobility_series[mobility_type] = np.empty(0)
    mobility_series[mobility_type + '_smooth'] = np.empty(0)
    if SAVE_PLOTS:
        os.makedirs(PLOT_PATH + 'mobility/' + mobility_type + '/', exist_ok=True)

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
    t0 = np.nan if len(data[data['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date']) == 0 else \
        data[data['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date'].iloc[0]
    t0_relative = np.nan if len(data[((data['confirmed']/population)*10000) >= POP_RELATIVE_T0_THRESHOLD]) == 0 else \
        data[((data['confirmed']/population)*10000) >= POP_RELATIVE_T0_THRESHOLD]['date'].iloc[0]
    t0_1_dead = np.nan if len(data[data['dead']>=1]['date']) == 0 else \
        data[data['dead']>=1]['date'].iloc[0]
    t0_5_dead = np.nan if len(data[data['dead']>=5]['date']) == 0 else \
        data[data['dead']>=5]['date'].iloc[0]
    t0_10_dead = np.nan if len(data[data['dead']>=10]['date']) == 0 else \
        data[data['dead']>=10]['date'].iloc[0]

    days_since_t0 = np.repeat(np.nan,len(data)) if pd.isnull(t0) else \
        np.array([(date - t0).days for date in data['date'].values])
    days_since_t0_relative = np.repeat(np.nan,len(data)) if pd.isnull(t0_relative) else \
        np.array([(date - t0_relative).days for date in data['date'].values])
    days_since_t0_1_dead = np.repeat(np.nan,len(data)) if pd.isnull(t0_1_dead) else \
        np.array([(date - t0_1_dead).days for date in data['date'].values])
    days_since_t0_5_dead = np.repeat(np.nan,len(data)) if pd.isnull(t0_5_dead) else \
        np.array([(date - t0_5_dead).days for date in data['date'].values])
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
    epidemiology_series['days_since_t0'] = np.concatenate(
        (epidemiology_series['days_since_t0'], days_since_t0))
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
    epidemiology_series['days_since_t0_pop'] = np.concatenate(
        (epidemiology_series['days_since_t0_pop'], days_since_t0_relative))
    epidemiology_series['days_since_t0_1_dead'] = np.concatenate(
        (epidemiology_series['days_since_t0_1_dead'], days_since_t0_1_dead))
    epidemiology_series['days_since_t0_5_dead'] = np.concatenate(
        (epidemiology_series['days_since_t0_5_dead'], days_since_t0_5_dead))
    epidemiology_series['days_since_t0_10_dead'] = np.concatenate(
        (epidemiology_series['days_since_t0_10_dead'], days_since_t0_10_dead))


'''
MOBILITY TIME SERIES PROCESSING
'''

countries = np.sort(mobility['countrycode'].unique())
for country in tqdm(countries, desc='Processing Mobility Time Series Data'):
    data = mobility[mobility['countrycode'] == country]

    mobility_series['countrycode'] = np.concatenate((
        mobility_series['countrycode'], data['countrycode'].values))
    mobility_series['country'] = np.concatenate(
        (mobility_series['country'], data['country'].values))
    mobility_series['date'] = np.concatenate(
        (mobility_series['date'], data['date'].values))

    for mobility_type in mobilities:
        #x = np.arange(len(data['date']))
        #y = data[mobility_type].values
        #ys = csaps(x, y, x, smooth=SMOOTH)
        ys = data[[mobility_type,'date']].rolling(window=7, on='date').mean()[mobility_type]

        mobility_series[mobility_type] = np.concatenate((
            mobility_series[mobility_type], data[mobility_type].values))
        mobility_series[mobility_type + '_smooth'] = np.concatenate((
            mobility_series[mobility_type + '_smooth'], ys))

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
mobility_series = pd.DataFrame.from_dict(mobility_series)
government_response_series = pd.DataFrame.from_dict(government_response_series)
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 2 - PROCESSING PANEL DATA & PLOTTING NEW CASES PER DAY 
'''

'''
EPI:
COUNTRYCODE - UNIQUE IDENTIFIER √
COUNTRY - FULL COUNTRY NAME √
CLASS - FIRST WAVE (ASCENDING - 1, DESCENDING - 2) OR SECOND WAVE (ASCENDING - 3, DESCENDING - 4) √
POPULATION √
T0 - DATE FIRST N CASES CONFIRMED √
T0 RELATIVE - DATE FIRST N CASES PER MILLION CONFIRMED √
PEAK_1 - HEIGHT OF FIRST PEAK √
PEAK_2 - HEIGHT OF SECOND PEAK √
PEAK_1_PER_10K
PEAK_2_PER_10K
PEAK_1_DEAD_PER_10K
PEAK_2_DEAD_PER_10K
DATE_PEAK_1 - DATE OF FIRST WAVE PEAK √
DATE_PEAK_2 - DATE OF SECOND WAVE PEAK
FIRST_WAVE_START - START OF FIRST WAVE  √
FIRST_WAVE_END - END OF FIRST WAVE √
SECOND_WAVE_START - START OF SECOND WAVE √
SECOND_WAVE_END - END OF SECOND WAVE √
LAST_CONFIRMED - LATEST NUMBER OF CONFIRMED CASES √
TESTING DATA AVAILABLE √
CASE FATALITY RATE PER WAVE 
PLOT ALL FOR LABELLING √
MOB:
PEAK_VALUE OF MOBILITY 
PEAK DATE OF MOBILITY
QUARANTINE FATIGUE
GOV:
COUNTRYCODE √
COUNTRY √
MAX SI - VALUE OF MAXIMUM SI √
DATE OF PEAK SI - √
RESPONSE TIME - TIME FROM T0 T0 PEAK SI √
RESPONSE TIME - TIME FROM T0_POP TO PEAK SI √
RESPONSE TIME - TIME FROM T0_POP TO FLAG RAISED 
FLAG_RAISED - DATE FLAG RAISED FOR EACH FLAG IN L66 √
FLAG_LOWERED - DATE FLAG LOWERED FOR EACH FLAG IN L66 √
FLAG_RASIED_AGAIN - DATE FLAG RAISED AGAIN FOR EACH FLAG IN L66 √
FLAG RESPONSE TIME - T0_POP TO MAX FLAG FOR EACH FLAG IN L66√
FLAG TOTAL DAYS √
'''

epidemiology_panel = pd.DataFrame(columns=['countrycode', 'country', 'class', 'population', 't0', 't0_relative',
                                           't0_1_dead','t0_5_dead','t0_10_dead',
                                           'peak_1', 'peak_2', 'date_peak_1', 'date_peak_2', 'first_wave_start',
                                           'first_wave_end', 'duration_first_wave', 'second_wave_start', 'second_wave_end','last_confirmed',
                                           'last_dead','testing_available','peak_1_cfr','peak_2_cfr',
                                           'dead_class','tests_class',
                                           'tau_dead','tau_p_dead','tau_tests','tau_p_tests',
                                           'confirmed_30d_avg_before_first_peak','dead_30d_avg_before_first_peak','tests_30d_avg_before_first_peak',
                                           'confirmed_30d_avg_before_second_peak','dead_30d_avg_before_second_peak','tests_30d_avg_before_second_peak'])

if SAVE_PLOTS: # Create directories for saving plots, remove any existing plots
    if os.path.isdir(PLOT_PATH + 'epidemiological/'):
            shutil.rmtree(PLOT_PATH + 'epidemiological/')
    for i in range(0,7):
        os.makedirs(PLOT_PATH + 'epidemiological/class_' + str(i) + '/', exist_ok=True)
    os.makedirs(PLOT_PATH + 'epidemiological/class_other' + '/', exist_ok=True)

# There are some countries with poor data quality in the ECDC data, insufficient for a clear classification
# They are manually labelled from visual inspection, and are defined here.
exclude_countries = ['CMR','COG','GNQ','BWA','ESH']
# Some countries also have poor data quality, but the overall shape of the wave is still visible:
# KGZ, CHL, NIC, KAZ, ECU, CHN

countries = epidemiology['countrycode'].unique()
for country in tqdm(countries, desc='Processing Epidemiological Panel Data'):
    data = dict()
    data['peak_1'] = np.nan
    data['peak_2'] = np.nan
    data['peak_1_per_10k'] = np.nan
    data['peak_2_per_10k'] = np.nan
    data['peak_1_dead_per_10k'] = np.nan
    data['peak_2_dead_per_10k'] = np.nan
    data['date_peak_1'] = np.nan
    data['date_peak_2'] = np.nan
    data['first_wave_start'] = np.nan
    data['first_wave_end'] = np.nan
    data['duration_first_wave'] = np.nan
    data['second_wave_start'] = np.nan
    data['second_wave_end'] = np.nan
    data['peak_1_cfr'] = np.nan
    data['peak_2_cfr'] = np.nan
    data['tau_dead'] = np.nan
    data['tau_p_dead'] = np.nan
    data['tau_tests'] = np.nan
    data['tau_p_tests'] = np.nan
    data['confirmed_30d_avg_before_first_peak'] = np.nan
    data['dead_30d_avg_before_first_peak'] = np.nan
    data['tests_30d_avg_before_first_peak'] = np.nan
    data['confirmed_30d_avg_before_second_peak'] = np.nan
    data['dead_30d_avg_before_second_peak'] = np.nan
    data['tests_30d_avg_before_second_peak'] = np.nan
    
    data['countrycode'] = country
    country_series = epidemiology_series[epidemiology_series['countrycode'] == country].reset_index(drop=True)
    if len(country_series) < DISTANCE: # If the time series does not have sufficient length, skip the country
        continue
    
    data['country'] = country_series['country'].iloc[0]
    data['population'] = np.nan if len(wb_statistics[wb_statistics['countrycode'] == country]) == 0 else \
        wb_statistics[wb_statistics['countrycode'] == country]['value'].values[0]
    data['t0'] = np.nan if len(country_series[country_series['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date']) == 0 else \
        country_series[country_series['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date'].iloc[0]
    data['t0_relative'] = np.nan if len(country_series[(country_series['confirmed'] / data['population'] * 10000
                                      >= POP_RELATIVE_T0_THRESHOLD)]['date']) == 0 else \
        country_series[(country_series['confirmed'] / data['population'] * 10000
                                      >= POP_RELATIVE_T0_THRESHOLD)]['date'].iloc[0]
    data['t0_1_dead'] = np.nan if len(country_series[country_series['dead']>=1]['date']) == 0 else \
        country_series[country_series['dead']>=1]['date'].iloc[0]
    data['t0_5_dead'] = np.nan if len(country_series[country_series['dead']>=5]['date']) == 0 else \
        country_series[country_series['dead']>=5]['date'].iloc[0]
    data['t0_10_dead'] = np.nan if len(country_series[country_series['dead']>=10]['date']) == 0 else \
        country_series[country_series['dead']>=10]['date'].iloc[0]

    cases_prominence_threshold = max(ABS_PROMINENCE_THRESHOLD,min(POP_PROMINENCE_THRESHOLD*data['population']/10000,
                                                                  ABS_PROMINENCE_THRESHOLD_UPPER))
    peak_characteristics = find_peaks(country_series['new_per_day_smooth'].values,
                                      prominence=cases_prominence_threshold, distance=DISTANCE)
    peak_locations = peak_characteristics[0]
    peak_prominences = peak_characteristics[1]['prominences']
    genuine_peaks = [p for i,p in enumerate(peak_locations) 
                     if peak_prominences[i] >= RELATIVE_PROMINENCE_THRESHOLD * country_series['new_per_day_smooth'].values[p]]
    
    # Label country wave status
    if country in exclude_countries:
        data['class'] = 0
        genuine_peaks = []
    else:
        data['class'] = 2*len(genuine_peaks)
    # Increase the class by 1 if the country is entering a new peak
    if data['class'] > 0 and genuine_peaks[-1]<len(country_series):
        # Entering a new peak if the end of the time series can meet the criteria to be defined as a peak itself without any further increase
        last_peak_date = country_series['date'].values[genuine_peaks[-1]]
        trough_value = min(country_series.loc[country_series['date']>last_peak_date,'new_per_day_smooth'])
        trough_date = country_series['date'][np.argmin(country_series.loc[country_series['date']>last_peak_date,'new_per_day_smooth'])]
        max_after_trough = np.nanmax(country_series.loc[country_series['date']>=trough_date,'new_per_day_smooth'])
        if max_after_trough-trough_value >= cases_prominence_threshold \
        and max_after_trough-trough_value >= RELATIVE_PROMINENCE_THRESHOLD * max_after_trough:
            data['class'] = data['class'] + 1
    elif data['class'] == 0 and country not in exclude_countries:
        if np.nanmax(country_series['new_per_day_smooth']) >= CLASS_1_THRESHOLD:
            data['class'] = data['class'] + 1
    
    if data['class'] == 1 and not pd.isnull(data['t0_1_dead']):
        data['first_wave_start'] = data['t0_1_dead']
        data['first_wave_end'] = max(country_series['date'])
    
    if data['class'] >= 2:
        data['peak_1'] = country_series['new_per_day_smooth'].values[genuine_peaks[0]]
        data['peak_1_per_10k'] = country_series['new_cases_per_10k'].values[genuine_peaks[0]]
        # peak_1_dead_per_10k is at the peak date of cases, whereas dead_peak_1_per_10k is at the peak date of deaths
        data['peak_1_dead_per_10k'] = country_series['new_deaths_per_10k'].values[genuine_peaks[0]]
        data['date_peak_1'] = country_series['date'].values[genuine_peaks[0]]
        bases = np.append(peak_characteristics[1]['left_bases'],peak_characteristics[1]['right_bases'])
        # for first wave left base, take the min of: T0, or the closest base to the left
        data['first_wave_start'] = min(data['t0_1_dead'],country_series['date'].values[max([b for b in bases if b<genuine_peaks[0]])]) \
                                    if not pd.isnull(data['t0_1_dead']) \
                                    else country_series['date'].values[max([b for b in bases if b<genuine_peaks[0]])]
        # for first wave right base, take the closest base to the right
        data['first_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[0]])]
        data['duration_first_wave'] = (data['first_wave_end'] - data['first_wave_start']).days
        data['peak_1_cfr'] = country_series[
                                 (country_series['date'] >= data['first_wave_start'] +
                                  datetime.timedelta(days=DEATH_LAG)) &
                                 (country_series['date'] <= data['first_wave_end'] +
                                  datetime.timedelta(days=DEATH_LAG))]['dead_per_day_smooth'].sum()/\
                             country_series[
                                 (country_series['date'] >= data['first_wave_start']) &
                                 (country_series['date'] <= data['first_wave_end'])]['new_per_day_smooth'].sum()
        data['dead_first_wave'] = country_series.loc[country_series['date']==data['first_wave_end'],'dead'].values[0] \
                                  - country_series.loc[country_series['date']==data['first_wave_start'],'dead'].values[0]

        data['dead_peak_1'] = country_series['dead_per_day_smooth'].values[genuine_peaks[0]]
    
    if data['class'] >= 3:
        data['second_wave_start'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[0]])]
        data['second_wave_end'] = max(country_series['date'])
        data['duration_second_wave'] = (data['second_wave_end'] - data['second_wave_start']).days

    if data['class'] >= 4:
        data['peak_2'] = country_series['new_per_day_smooth'].values[genuine_peaks[1]]
        data['peak_2_per_10k'] = country_series['new_cases_per_10k'].values[genuine_peaks[1]]
        data['peak_2_dead_per_10k'] = country_series['new_deaths_per_10k'].values[genuine_peaks[1]]
        data['date_peak_2'] = country_series['date'].values[genuine_peaks[1]]
        data['second_wave_start'] = country_series['date'].values[max([b for b in bases if b<genuine_peaks[1]])]
        data['second_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[1]])]
        data['peak_2_cfr'] = country_series[
                                 (country_series['date'] >= data['second_wave_start'] +
                                  datetime.timedelta(days=DEATH_LAG)) &
                                 (country_series['date'] <= data['second_wave_end'] +
                                  datetime.timedelta(days=DEATH_LAG))]['dead_per_day_smooth'].sum()/\
                             country_series[
                                 (country_series['date'] >= data['second_wave_start']) &
                                 (country_series['date'] <= data['second_wave_end'])]['new_per_day_smooth'].sum()
        data['dead_second_wave'] = country_series.loc[country_series['date']==data['second_wave_end'],'dead'].values[0] \
                                  - country_series.loc[country_series['date']==data['second_wave_start'],'dead'].values[0]
        data['dead_peak_2'] = country_series['dead_per_day_smooth'].values[genuine_peaks[1]]
    data['last_confirmed'] = country_series['confirmed'].iloc[-1]
    data['last_dead'] = country_series['dead'].iloc[-1]
    data['testing_available'] = True if len(country_series['new_tests'].dropna()) > 0 else False

    # Classify wave status for deaths
    dead_prominence_threshold = max(ABS_PROMINENCE_THRESHOLD_DEAD, min(POP_PROMINENCE_THRESHOLD_DEAD * data['population'] / 10000,
                                                                       ABS_PROMINENCE_THRESHOLD_UPPER_DEAD))
    dead_peak_characteristics = find_peaks(country_series['dead_per_day_smooth'].values,
                                      prominence=dead_prominence_threshold, distance=DISTANCE)
    peak_locations = dead_peak_characteristics[0]
    peak_prominences = dead_peak_characteristics[1]['prominences']
    dead_genuine_peaks = [p for i,p in enumerate(peak_locations) 
                     if peak_prominences[i] >= RELATIVE_PROMINENCE_THRESHOLD * country_series['dead_per_day_smooth'].values[p]]

    # Label class for deaths
    if country in exclude_countries:
        data['dead_class'] = 0
        dead_genuine_peaks = []
    else:
        data['dead_class'] = 2*len(dead_genuine_peaks)
    
    if data['dead_class'] > 0 and dead_genuine_peaks[-1]<len(country_series):
        last_peak_date = country_series['date'].values[dead_genuine_peaks[-1]]
        trough_value = min(country_series.loc[country_series['date']>last_peak_date,'dead_per_day_smooth'])
        trough_date = country_series['date'][np.argmin(country_series.loc[country_series['date']>last_peak_date,'dead_per_day_smooth'])]
        max_after_trough = np.nanmax(country_series.loc[country_series['date']>=trough_date,'dead_per_day_smooth'])
        if max_after_trough-trough_value >= dead_prominence_threshold \
        and max_after_trough-trough_value >= RELATIVE_PROMINENCE_THRESHOLD * max_after_trough:
            data['dead_class'] = data['dead_class'] + 1
    elif data['dead_class'] == 0 and country not in exclude_countries:
        if np.nanmax(country_series['dead_per_day_smooth']) >= CLASS_1_THRESHOLD_DEAD:
            data['dead_class'] = data['dead_class'] + 1
    
    if len(dead_genuine_peaks) >= 1:
        data['date_dead_peak_1'] = country_series['date'].values[dead_genuine_peaks[0]]
        bases = np.append(dead_peak_characteristics[1]['left_bases'],dead_peak_characteristics[1]['right_bases'])
        data['dead_first_wave_start'] = min(data['t0_1_dead'],country_series['date'].values[max([b for b in bases if b<dead_genuine_peaks[0]])]) \
                                        if not pd.isnull(data['t0_1_dead']) \
                                        else country_series['date'].values[max([b for b in bases if b<dead_genuine_peaks[0]])]
        data['dead_first_wave_end'] = country_series['date'].values[min([b for b in bases if b>dead_genuine_peaks[0]])]
    if len(dead_genuine_peaks) >= 2:
        data['date_dead_peak_2'] = country_series['date'].values[dead_genuine_peaks[1]]
        data['dead_second_wave_start'] = country_series['date'].values[max([b for b in bases if b<dead_genuine_peaks[1]])]
        data['dead_second_wave_end'] = country_series['date'].values[min([b for b in bases if b>dead_genuine_peaks[1]])]
    # Classify wave status for tests
    if data['testing_available']:
        # Get kendall's correlation tau: Confirmed on Deaths (forward 14 days), vs. Confirmed on Tests. If tests explains more of the variance, may indicate 2nd wave is test driven.
        X = country_series['dead_per_day']
        y = country_series['new_per_day']
        X = np.array(X.iloc[DEATH_LAG:]).reshape(-1,1)     # X day lag from confirmed to death
        y = y.iloc[:-DEATH_LAG]
        data['tau_dead'],data['tau_p_dead'] = kendalltau(X,y)
        X = country_series[['new_per_day','new_tests']].dropna(how='any')
        y = X['new_per_day']
        X = np.array(X['new_tests']).reshape(-1,1)     # assume no lag from test to confirmed
        data['tau_tests'],data['tau_p_tests'] = kendalltau(X,y)
        
    # Calculate average daily cases, deaths and tests during 30 day period before first and second wave
    if data['testing_available']:
        start_date = min(country_series[['date','confirmed','dead','new_tests']].dropna()['date'])
        end_date = max(country_series[['date','confirmed','dead','new_tests']].dropna()['date'])
    else: # testing data not available
        start_date = min(country_series[['date','confirmed','dead']].dropna()['date'])
        end_date = max(country_series[['date','confirmed','dead']].dropna()['date'])
    # For first wave, take the 30 days before the first peak, or if no first peak then the last 30 days of time series
    # Mean requires at least 14 days in the period
    if data['class'] == 1: 
        first = country_series.loc[(country_series['date']>=start_date) &
                                   (country_series['date']<=end_date) &
                                   (country_series['date']>=end_date-datetime.timedelta(days=30)),
                                   ['date','confirmed','dead','new_tests']]
    elif data['class'] >= 2:
        first = country_series.loc[(country_series['date']>=start_date) &
                                   (country_series['date']<=end_date) &
                                   (country_series['date']<=data['date_peak_1']) &
                                   (country_series['date']>=data['date_peak_1']-datetime.timedelta(days=30)),
                                   ['date','confirmed','dead','new_tests']]
    if data['class'] >= 1 and len(first) >= 14:
        data['confirmed_30d_avg_before_first_peak'] = (first.iloc[-1]['confirmed'] - first.iloc[0]['confirmed']) / len(first)
        data['dead_30d_avg_before_first_peak'] = (first.iloc[-1]['dead'] - first.iloc[0]['dead']) / len(first)
        if data['testing_available']:
            data['tests_30d_avg_before_first_peak'] = np.nanmean(first['new_tests'])
     # For second wave, take the 30 days before the second peak, or if entering second wwave then the last 30 days of time series
    if data['class'] == 3: 
        second = country_series.loc[(country_series['date']>=start_date) &
                                   (country_series['date']<=end_date) &
                                   (country_series['date']>=end_date-datetime.timedelta(days=30)),
                                   ['date','confirmed','dead','new_tests']]
    elif data['class'] >= 4:
        second = country_series.loc[(country_series['date']>=start_date) &
                                   (country_series['date']<=end_date) &
                                   (country_series['date']<=data['date_peak_2']) &
                                   (country_series['date']>=data['date_peak_2']-datetime.timedelta(days=30)),
                                   ['date','confirmed','dead','new_tests']]
    if data['class'] >= 3 and len(second) >= 14:
        data['confirmed_30d_avg_before_second_peak'] = (second.iloc[-1]['confirmed'] - second.iloc[0]['confirmed']) / len(second)
        data['dead_30d_avg_before_second_peak'] = (second.iloc[-1]['dead'] - second.iloc[0]['dead']) / len(second)
        if data['testing_available']:
            data['tests_30d_avg_before_second_peak'] = np.nanmean(second['new_tests'])
    
    if SAVE_PLOTS:
        f, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        axes[0].plot(country_series['date'].values,
                 country_series['new_per_day'].values,
                 label='New Cases per Day')
        axes[0].plot(country_series['date'].values,
                 country_series['new_per_day_smooth'].values,
                 label='New Cases per Day 7 Day Moving Average')
        axes[0].plot([country_series['date'].values[i]
                  for i in genuine_peaks],
                 [country_series['new_per_day_smooth'].values[i]
                  for i in genuine_peaks], "X", ms=20, color='red')
        axes[1].plot(country_series['date'].values,
                 country_series['dead_per_day'].values,
                 label='Deaths per Day')
        axes[1].plot(country_series['date'].values,
                 country_series['dead_per_day_smooth'].values,
                 label='Deaths per Day 7 Day Moving Average')
        axes[1].plot([country_series['date'].values[i]
                  for i in dead_genuine_peaks],
                 [country_series['dead_per_day_smooth'].values[i]
                  for i in dead_genuine_peaks], "X", ms=20, color='red')
        if data['testing_available']:
            axes[2].plot(country_series['date'].values,
                     country_series['new_tests'].values,
                     label='Tests per Day')
            axes[2].plot(country_series['date'].values,
                     country_series['new_tests_smooth'].values,
                     label='Tests per Day 7 Day Moving Average')
        axes[0].set_title('New Cases per Day')
        axes[0].set_ylabel('New Cases per Day')
        axes[1].set_title('Deaths per Day')
        axes[1].set_ylabel('Deaths per Day')
        axes[2].set_title('Tests per Day')
        axes[2].set_ylabel('Tests per Day')
        f.suptitle('Cases, Deaths and Tests per Day for ' + data['country'])
        if data['class'] <= 6:
            plt.savefig(PLOT_PATH + 'epidemiological/class_' +str(data['class']) +'/' + data['country'] + '.png')
        else:
            plt.savefig(PLOT_PATH + 'epidemiological/class_other' + '/' + data['country'] + '.png')
        plt.close('all')    
    
    epidemiology_panel = epidemiology_panel.append(data,ignore_index=True)


mobility_panel = pd.DataFrame(columns=['countrycode','country'] +
                                      [mobility_type + '_max' for mobility_type in mobilities] +
                                      [mobility_type + '_max_date' for mobility_type in mobilities] +
                                      [mobility_type + '_quarantine_fatigue' for mobility_type in mobilities] + 
                                      [mobility_type + '_integral' for mobility_type in mobilities])

countries = mobility['countrycode'].unique()
for country in tqdm(countries, desc='Processing Mobility Panel Data'):
    data = dict()
    data['countrycode'] = country
    country_series = mobility_series[mobility_series['countrycode']==country].reset_index(drop=True)
    data['country'] = country_series['country'].iloc[0]
    for mobility_type in mobilities:
        if sum(~np.isnan(country_series[mobility_type])) > 1:
            data[mobility_type + '_max'] = country_series[mobility_type + '_smooth'].max()
            data[mobility_type + '_max_date'] = country_series.iloc[country_series[mobility_type + '_smooth'].argmax()]['date']
            data[mobility_type + '_quarantine_fatigue'] = np.nan
            data[mobility_type + '_integral'] = np.trapz(y=country_series[mobility_type].dropna(), 
                                                         x=[(a-country_series['date'].values[0]).days for a in country_series['date'][~np.isnan(country_series[mobility_type])]])

            mob_data_to_fit = country_series[mobility_type + '_smooth'].iloc[country_series[mobility_type + '_smooth'].argmax()::]
            mob_data_to_fit = mob_data_to_fit.dropna()
            if len(mob_data_to_fit) != 0:
                data[mobility_type + '_quarantine_fatigue'] = LinearRegression().fit(
                    np.arange(len(mob_data_to_fit)).reshape(-1,1),mob_data_to_fit.values).coef_[0]

    mobility_panel = mobility_panel.append(data, ignore_index=True)

    
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
    t0 = np.nan if len(epidemiology_panel[epidemiology_panel['countrycode']==country]['t0_1_dead']) == 0 \
        else epidemiology_panel[epidemiology_panel['countrycode']==country]['t0_1_dead'].iloc[0]
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
        if not pd.isnull(t0):
            # SI value at T0
            data['si_at_t0'] = country_series.loc[country_series['date']==t0,'si'].values[0]
            # days taken to reach threshold
            data['si_days_to_threshold'] = (min(country_series.loc[country_series['si']>=SI_THRESHOLD,'date']) - t0).days \
                                            if sum(country_series['si']>=SI_THRESHOLD) > 0 else np.nan
            for flag in flags:
                 data[flag + '_at_t0'] = country_series.loc[country_series['date']==t0,flag].values[0]
                 data[flag + '_days_to_threshold'] = (min(country_series.loc[country_series[flag]>=flag_thresholds[flag],'date']) - t0).days \
                                                      if sum(country_series[flag]>=flag_thresholds[flag]) > 0 else np.nan
        if not (pd.isnull(date_peak_1) or pd.isnull(first_wave_start) or pd.isnull(first_wave_end)):
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


# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 3 - FIGURE 1
'''
# ['countrycode','country','days_to_t0','days_to_t0_pop']
start_date = epidemiology_series['date'].min()
figure_1 = pd.DataFrame(columns=['countrycode', 'country', 'days_to_t0', 'days_to_t0_5_dead', 'days_to_t0_1_dead', 'days_to_t0_10_dead', 'class',
                                 'new_cases_per_10k', 'new_deaths_per_10k','geometry'])
figure_1 = epidemiology_series[['countrycode', 'country','new_cases_per_10k', 'new_deaths_per_10k']]
figure_1 = figure_1.merge(epidemiology_panel[['countrycode', 't0_relative', 't0_1_dead','t0_5_dead', 't0_10_dead', 'class']], on=['countrycode'], how='left')
figure_1['days_to_t0'] = (figure_1['t0_relative']-start_date).apply(lambda x: x.days)
figure_1['days_to_t0_1_dead'] = (figure_1['t0_1_dead']-start_date).apply(lambda x: x.days)
figure_1['days_to_t0_5_dead'] = (figure_1['t0_5_dead']-start_date).apply(lambda x: x.days)
figure_1['days_to_t0_10_dead'] = (figure_1['t0_10_dead']-start_date).apply(lambda x: x.days)

figure_1 = figure_1.drop(columns=['t0_relative','t0_1_dead','t0_5_dead', 't0_10_dead'])
figure_1 = figure_1.merge(map_data[['countrycode','geometry']], on=['countrycode'], how='left')

if SAVE_CSV:
    #figure_1.to_csv(CSV_PATH + 'figure_1.csv', sep=';')
    figure_1.astype({'geometry': str}).to_csv(CSV_PATH + 'figure_1.csv', sep=';')
    
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 4 - FIGURE 2
'''

figure_2 = epidemiology_series[['country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
    'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smooth', 'positive_rate','positive_rate_smooth']]

if SAVE_CSV:
    figure_2.to_csv(CSV_PATH + 'figure_2.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 5 - FIGURE 3
'''
class_coarse = {
    0:'EPI_OTHER',
    1:'EPI_FIRST_WAVE',
    2:'EPI_FIRST_WAVE',
    3:'EPI_SECOND_WAVE',
    4:'EPI_SECOND_WAVE',
    5:'EPI_THIRD_WAVE',
    6:'EPI_THIRD_WAVE'
}

# Figure 3a: Scatterplot of cumulative deaths per 10k population (integral of deaths curve) against integral of SI curve
data = epidemiology_panel[['countrycode','country','class','last_confirmed','last_dead']].merge(
            government_response_panel[['countrycode','si_integral']],on='countrycode',how='left').merge(
                mobility_panel[['countrycode'] + [mobility_type+'_integral' for mobility_type in mobilities]],on='countrycode',how='left')
data['class_coarse'] = [class_coarse[x] if x in class_coarse.keys() else 'EPI_OTHER' for x in data['class'].values]
data['last_confirmed_per_10k'] = 10000 * epidemiology_panel['last_confirmed'] / epidemiology_panel['population']
data['last_dead_per_10k'] = 10000 * epidemiology_panel['last_dead'] / epidemiology_panel['population']

figure_3a = data

if SAVE_CSV:
    figure_3a.to_csv(CSV_PATH + 'figure_3a.csv')

# Figure 3b: Scatterplot of cumulative deaths per 10k population against  government response time
data = epidemiology_panel[['countrycode', 'country', 'class' , 'population', 'last_confirmed','last_dead',
                           't0_relative','t0_1_dead','t0_5_dead','t0_10_dead',
                           'duration_first_wave','duration_second_wave',
                           'dead_first_wave', 'dead_second_wave']]
data['class_coarse'] = [class_coarse[x] if x in class_coarse.keys() else 'EPI_OTHER' for x in data['class'].values]
data['last_confirmed_per_10k'] = 10000 * epidemiology_panel['last_confirmed'] / epidemiology_panel['population']
data['last_dead_per_10k'] = 10000 * epidemiology_panel['last_dead'] / epidemiology_panel['population']
data = data.merge(government_response_panel[['countrycode','si_days_to_max_si','max_si'] +
                                            [flag + '_at_t0' for flag in flags + ['si']] +
                                            [flag + '_at_peak_1' for flag in flags + ['si']] +
                                            [flag + '_days_to_threshold' for flag in flags + ['si']] +
                                            [flag + '_days_above_threshold_first_wave' for flag in flags + ['si']]],
                  how='left', on='countrycode')

figure_3b = pd.DataFrame(columns=['COUNTRYCODE', 'COUNTRY', 'GOV_MAX_SI_DAYS_FROM_T0',
                                 'CLASS_COARSE', 'POPULATION', 'EPI_CONFIRMED', 'EPI_CONFIRMED_PER_10K',
                                 'EPI_DEAD', 'EPI_DEAD_PER_10K','MAX_SI',
                                 'T0_POP','T0_1_DEAD','T0_5_DEAD','T0_10_DEAD',
                                 'EPI_DURATION_FIRST_WAVE','EPI_DURATION_SECOND_WAVE',
                                 'DEAD_FIRST_WAVE', 'DEAD_SECOND_WAVE'] +
                                 [flag.upper() + '_AT_T0' for flag in flags + ['si']] +
                                 [flag.upper() + '_AT_PEAK_1' for flag in flags + ['si']] +
                                 [flag.upper() + '_DAYS_TO_THRESHOLD' for flag in flags + ['si']] +
                                 [flag.upper() + '_DAYS_ABOVE_THRESHOLD_FIRST_WAVE' for flag in flags + ['si']])

figure_3b['COUNTRYCODE'] = data['countrycode']
figure_3b['COUNTRY'] = data['country']
figure_3b['GOV_MAX_SI_DAYS_FROM_T0'] = data['si_days_to_max_si']
figure_3b['CLASS'] = data['class']
figure_3b['CLASS_COARSE'] = data['class_coarse']
figure_3b['POPULATION'] = data['population']
figure_3b['EPI_CONFIRMED'] = data['last_confirmed']
figure_3b['EPI_CONFIRMED_PER_10K'] = data['last_confirmed_per_10k']
figure_3b['EPI_DEAD'] = data['last_dead']
figure_3b['EPI_DEAD_PER_10K'] = data['last_dead_per_10k']
figure_3b['MAX_SI'] = data['max_si']
figure_3b['T0_POP'] = data['t0_relative']
figure_3b['T0_1_DEAD'] = data['t0_1_dead']
figure_3b['T0_5_DEAD'] = data['t0_5_dead']
figure_3b['T0_10_DEAD'] = data['t0_10_dead']
figure_3b['EPI_DURATION_FIRST_WAVE'] = data['duration_first_wave']
figure_3b['EPI_DURATION_SECOND_WAVE'] = data['duration_second_wave']
figure_3b['DEAD_FIRST_WAVE'] = data['dead_first_wave']
figure_3b['DEAD_SECOND_WAVE'] = data['dead_second_wave']
for flag in flags + ['si']:
    figure_3b[flag.upper() + '_AT_T0'] = data[flag + '_at_t0']
    figure_3b[flag.upper() + '_AT_PEAK_1'] = data[flag + '_at_peak_1']
    figure_3b[flag.upper() + '_DAYS_TO_THRESHOLD'] = data[flag + '_days_to_threshold']
    figure_3b[flag.upper() + '_DAYS_ABOVE_THRESHOLD_FIRST_WAVE'] = data[flag + '_days_above_threshold_first_wave']

if SAVE_CSV:
    figure_3b.to_csv(CSV_PATH + 'figure_3b.csv')


# Figure 3b response time scatterplot - data at the wave level
figure_3b_wave_level = pd.DataFrame()

countries = epidemiology_panel['countrycode'].unique()
for country in tqdm(countries, desc='Processing figure 3b data'):       
       data = dict()
       data['si_at_t0_1_dead'] = np.nan       
       data['si_at_t0_5_dead'] = np.nan
       data['si_at_t0_10_dead'] = np.nan       
       data['first_date_si_above_threshold'] = np.nan
       for flag in flags:
           data['first_date_'+flag[0:2]+'_above_threshold'] = np.nan
       data['response_time_t0_1_dead'] = np.nan       
       data['response_time_t0_5_dead'] = np.nan
       data['response_time_t0_10_dead'] = np.nan
       data['dead_during_wave'] = np.nan
       data['confirmed_during_wave'] = np.nan
       data['countrycode'] = country
       data['country'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'country'].values[0]
       data['class'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'class'].values[0]
       data['population'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'population'].values[0]
       if data['class'] >= 1:
              # First wave
              data['wave'] = 1
              data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_start'].values[0]
              data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_end'].values[0]
              data['t0_1_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_1_dead'].values[0]
              data['t0_5_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_5_dead'].values[0]
              data['t0_10_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_10_dead'].values[0]
             
              gov_country_series = government_response_series[government_response_series['countrycode']==country]
              country_series = epidemiology_series[epidemiology_series['countrycode']==country]
              if len(gov_country_series)==0 or len(country_series)==0:
                  continue
                
              if ~pd.isnull(data['t0_1_dead']) and data['t0_1_dead'] in gov_country_series['date'].values:
                      data['si_at_t0_1_dead'] = gov_country_series.loc[gov_country_series['date']==data['t0_1_dead'],'si'].values[0]
              if ~pd.isnull(data['t0_5_dead']) and data['t0_5_dead'] in gov_country_series['date'].values:
                      data['si_at_t0_5_dead'] = gov_country_series.loc[gov_country_series['date']==data['t0_5_dead'],'si'].values[0]
              if ~pd.isnull(data['t0_10_dead']) and data['t0_10_dead'] in gov_country_series['date'].values:
                      data['si_at_t0_10_dead'] = gov_country_series.loc[gov_country_series['date']==data['t0_10_dead'],'si'].values[0]
              if sum(gov_country_series['si']>=SI_THRESHOLD) > 0:
                      data['first_date_si_above_threshold'] = min(gov_country_series.loc[gov_country_series['si']>=SI_THRESHOLD,'date'])
              for flag in flags:
                  if sum(gov_country_series[flag]>=flag_thresholds[flag]) > 0:
                      data['first_date_'+flag[0:2]+'_above_threshold'] = min(gov_country_series.loc[gov_country_series[flag]>=flag_thresholds[flag],'date'])
              data['dead_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'dead'].values[0] - \
                                         country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]
              data['confirmed_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'confirmed'].values[0] - \
                                              country_series.loc[country_series['date']==data['wave_start'],'confirmed'].values[0]
 
              figure_3b_wave_level = figure_3b_wave_level.append(data, ignore_index=True)
 
              if data['class'] >= 3:
                      # Second wave
                      data['t0_1_dead'] = np.nan
                      data['t0_5_dead'] = np.nan
                      data['t0_10_dead'] = np.nan
                      data['si_at_t0_1_dead'] = np.nan
                      data['si_at_t0_5_dead'] = np.nan
                      data['si_at_t0_10_dead'] = np.nan
                      data['response_time_t0_1_dead'] = np.nan
                      data['response_time_t0_5_dead'] = np.nan
                      data['response_time_t0_10_dead'] = np.nan
                      data['dead_during_wave'] = np.nan
                      data['confirmed_during_wave'] = np.nan
 
                      data['wave'] = 2
                      data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'second_wave_start'].values[0]
                      data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'second_wave_end'].values[0]

                      dead_at_start = country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]
                      data['t0_1_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                             (country_series['date']<=data['wave_end']) & \
                                                             (country_series['dead'] >= dead_at_start + 1), \
                                                             'date']
                      if len(data['t0_1_dead']) > 0:
                          data['t0_1_dead'] = data['t0_1_dead'].values[0]
                      else:
                          data['t0_1_dead'] = np.nan
                      data['t0_5_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                             (country_series['date']<=data['wave_end']) & \
                                                             (country_series['dead'] >= dead_at_start + 5), \
                                                             'date']
                      if len(data['t0_5_dead']) > 0:
                          data['t0_5_dead'] = data['t0_5_dead'].values[0]
                      else:
                          data['t0_5_dead'] = np.nan
                      data['t0_10_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                              (country_series['date']<=data['wave_end']) & \
                                                              (country_series['dead'] >= dead_at_start + 10), \
                                                              'date']
                      if len(data['t0_10_dead']) > 0:
                          data['t0_10_dead'] = data['t0_10_dead'].values[0]
                      else:
                          data['t0_10_dead'] = np.nan
                          
                      if ~pd.isnull(data['t0_1_dead']) and data['t0_1_dead'] in gov_country_series['date'].values:
                             data['si_at_t0_1_dead'] = gov_country_series.loc[gov_country_series['date']==data['t0_1_dead'],'si'].values[0]
                      if ~pd.isnull(data['t0_5_dead']) and data['t0_5_dead'] in gov_country_series['date'].values:
                             data['si_at_t0_5_dead'] = gov_country_series.loc[gov_country_series['date']==data['t0_5_dead'],'si'].values[0]
                      if ~pd.isnull(data['t0_10_dead']) and data['t0_10_dead'] in gov_country_series['date'].values:
                             data['si_at_t0_10_dead'] = gov_country_series.loc[gov_country_series['date']==data['t0_10_dead'],'si'].values[0]
                             
                      data['dead_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'dead'].values[0] - \
                                                 country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]
                      data['confirmed_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'confirmed'].values[0] - \
                                                      country_series.loc[country_series['date']==data['wave_start'],'confirmed'].values[0]

                      figure_3b_wave_level = figure_3b_wave_level.append(data, ignore_index=True)


if SAVE_CSV:
    figure_3b_wave_level.to_csv(CSV_PATH + 'figure_3b_wave_level.csv')

# Figure 3c: scatterplot of early testing against deaths
# 3c requires 2 data sets: 
# 1) Country data at the wave level
# 2) Testing at the country-day level
figure_3c_1 = pd.DataFrame()
countries = epidemiology_panel['countrycode'].unique()
for country in tqdm(countries, desc='Processing figure 3c data'):       
       data = dict()
       data['dead_during_wave'] = np.nan
       data['countrycode'] = country
       data['country'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'country'].values[0]
       data['class'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'class'].values[0]
       data['population'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'population'].values[0]
       if data['class'] >= 1:
              # First wave
              data['wave'] = 1
              data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_start'].values[0]
              data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'first_wave_end'].values[0]
              data['t0_1_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_1_dead'].values[0]
              data['t0_5_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_5_dead'].values[0]
              data['t0_10_dead'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'t0_10_dead'].values[0]
             
              country_series = epidemiology_series[epidemiology_series['countrycode']==country]

              data['dead_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'dead'].values[0] - \
                                         country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]

              figure_3c_1 = figure_3c_1.append(data, ignore_index=True)
 
              if data['class'] >= 3:
                      # Second wave
                      data['t0_1_dead'] = np.nan
                      data['t0_5_dead'] = np.nan
                      data['t0_10_dead'] = np.nan
                      data['dead_during_wave'] = np.nan
                      data['wave'] = 2
                      data['wave_start'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'second_wave_start'].values[0]
                      data['wave_end'] = epidemiology_panel.loc[epidemiology_panel['countrycode']==country,'second_wave_end'].values[0]

                      dead_at_start = country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]
                      data['t0_1_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                             (country_series['date']<=data['wave_end']) & \
                                                             (country_series['dead'] >= dead_at_start + 1), \
                                                             'date']
                      if len(data['t0_1_dead']) > 0:
                          data['t0_1_dead'] = data['t0_1_dead'].values[0]
                      else:
                          data['t0_1_dead'] = np.nan
                      data['t0_5_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                             (country_series['date']<=data['wave_end']) & \
                                                             (country_series['dead'] >= dead_at_start + 5), \
                                                             'date']
                      if len(data['t0_5_dead']) > 0:
                          data['t0_5_dead'] = data['t0_5_dead'].values[0]
                      else:
                          data['t0_5_dead'] = np.nan
                      data['t0_10_dead'] = country_series.loc[(country_series['date']>data['wave_start']) & \
                                                              (country_series['date']<=data['wave_end']) & \
                                                              (country_series['dead'] >= dead_at_start + 10), \
                                                              'date']
                      if len(data['t0_10_dead']) > 0:
                          data['t0_10_dead'] = data['t0_10_dead'].values[0]
                      else:
                          data['t0_10_dead'] = np.nan
               
                      data['dead_during_wave'] = country_series.loc[country_series['date']==data['wave_end'],'dead'].values[0] - \
                                                 country_series.loc[country_series['date']==data['wave_start'],'dead'].values[0]

                      figure_3c_1 = figure_3c_1.append(data, ignore_index=True)

if SAVE_CSV:
    figure_3c_1.to_csv(CSV_PATH + '/figure_3c_1.csv')
    
    
figure_3c_2 = epidemiology_series.loc[~pd.isnull(epidemiology_series['tests']),
                                     ['countrycode','date','tests']]
# for each country with a late start date for tests data and starts with <=1000 tests,
# linear interpolate to 0 to match the confirmed cases time series
# go back to the first date of the confirmed time series 
figure_3c_2 = pd.DataFrame(columns=['countrycode','date','tests'])
for country in tqdm(epidemiology_series.countrycode.unique()):
    data = epidemiology_series.loc[epidemiology_series['countrycode']==country,['countrycode','date','tests']]
    if np.all(pd.isnull(data['tests'])):
        continue
    else:
        min_date = min(data['date'])
        min_tests = np.nanmin(data['tests'])
        if pd.isnull(data.loc[data['date']==min_date,'tests'].values[0]) \
        and min_tests <= 1000:
            data.loc[data['date']==min_date,'tests'] = 0 
            data['tests'] = data['tests'].interpolate(method='linear')
        else:
            data = data.loc[~pd.isnull(data['tests']),]
        figure_3c_2 = figure_3c_2.append(data, ignore_index=True)

if SAVE_CSV:
    figure_3c_2.to_csv(CSV_PATH + '/figure_3c_2.csv')

# -------------------------------------------------------------------------------------------------------------------- #
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
# -------------------------------------------------------------------------------------------------------------------- #
  
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


# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 7 - FIGURE 4b LASAGNA PLOT
'''
usa_t0 = epidemiology_panel[epidemiology_panel['countrycode'] == 'USA']['t0_relative'].iloc[0]
map_discrete_step = 5
figure_4b = usa_cases.merge(translation_csv[['input_adm_area_1', 'input_adm_area_2', 'gid']],
            left_on=['state', 'county'], right_on=['input_adm_area_1', 'input_adm_area_2'], how='left').merge(
            usa_populations[['FIPS', 'Lat', 'Long_']], left_on=['fips'], right_on=['FIPS'], how='left')

figure_4b = figure_4b[['fips', 'Long_', 'date', 'cases']]
figure_4b['date'] = (figure_4b['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d').date())-usa_t0).apply(lambda x:x.days)
figure_4b = figure_4b.sort_values\
    (by=['Long_', 'date', 'fips'], ascending=[True, True, True]).dropna(subset=['fips', 'Long_'])

figure_4b['long_discrete'] = figure_4b['Long_'].apply(lambda x: map_discrete_step * round(x / map_discrete_step))
figure_4b = figure_4b[figure_4b['long_discrete'] >= -125]

figure_4b = figure_4b.set_index(['fips','date'])
figure_4b = figure_4b.sort_index()
figure_4b['new_cases_per_day'] = np.nan
for idx in figure_4b.index.levels[0]:
    figure_4b['new_cases_per_day'][idx] = figure_4b['cases'][idx].diff()
figure_4b = figure_4b.reset_index()[['date', 'long_discrete', 'new_cases_per_day', 'fips']]
figure_4b['new_cases_per_day'] = figure_4b['new_cases_per_day'].clip(lower=0, upper=None)

heatmap = figure_4b[['date', 'long_discrete', 'new_cases_per_day']].groupby(
    by=['date', 'long_discrete'], as_index=False).sum()
heatmap = heatmap[heatmap['date'] >= 0]
bins = np.arange(heatmap['long_discrete'].min(),
                 heatmap['long_discrete'].max() + map_discrete_step, step=map_discrete_step)

emptyframe = pd.DataFrame(columns=['date', 'long_discrete', 'new_cases_per_day'])
emptyframe['date'] = np.repeat(np.unique(heatmap['date']), len(bins))
emptyframe['long_discrete'] = np.tile(bins, len(np.unique(heatmap['date'])))
emptyframe['new_cases_per_day'] = np.zeros(len(emptyframe['long_discrete']))
figure_4b = emptyframe.merge(
    heatmap, on=['date', 'long_discrete'], how='left', suffixes=['_', '']).fillna(0)[
    ['date', 'long_discrete', 'new_cases_per_day']]

figure_4b = pd.pivot_table(figure_4b,index=['date'],columns=['long_discrete'],values=['new_cases_per_day'])
figure_4b = figure_4b.reindex(index=figure_4b.index[::-1]).values.astype(int)
figure_4b = resize(figure_4b, (150, 150))

plt.figure(figsize=(10,10))
plt.imshow(figure_4b, cmap='plasma', interpolation='nearest')
plt.ylabel('Days Since T0')
plt.xlabel('Longitude')
plt.xticks([0, 75, 150], [-125, -95, -65])
plt.yticks([0, 75, 150], [186, 93, 0])
plt.savefig(PLOT_PATH + 'lasgna.png')
# -------------------------------------------------------------------------------------------------------------------- #
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
     'peak_2_dead_per_10k', 'peak_1_cfr', 'peak_2_cfr', 'first_wave_start', 'first_wave_end', 'second_wave_start',
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
table1['cfr_peak_1'] = data['peak_1_cfr']
table1['cfr_peak_2'] = data['peak_2_cfr']
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
