import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import os
import warnings
from tqdm import tqdm
import datetime
from skimage.transform import resize

from csaps import csaps
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
DISTANCE = 21
PROMINENCE_THRESHOLD = 5            # Absolute prominence threshold (in number of new cases)
PROMINENCE_THRESHOLD_DEAD = 2     # Absolute prominence threshold (in number of new deaths)
PROMINENCE_THRESHOLD_TESTS = 10      # Absolute prominence threshold (in number of new tests)
RELATIVE_PROMINENCE_THRESHOLD = 0.25 # Prominence relative to the max of time series (smoothed)
CLASS_1_THRESHOLD = 100             # Threshold in number of new cases per day (smoothed) to be considered entering first wave
CLASS_1_THRESHOLD_DEAD = 5          # Threshold in number of dead per day (smoothed) to be considered entering first wave for deaths
CLASS_1_THRESHOLD_TESTS = 200       # Threshold in number of tests per day (smoothed) to be considered entering first wave for tests
ABSOLUTE_T0_THRESHOLD = 1000
POP_RELATIVE_T0_THRESHOLD = 5 #per million people
TEST_LAG = 0 # Lag between test date and test results
DEATH_LAG = 21 # Lag between confirmed and death. Ideally would be sampled from a random distribution of some sorts
SI_THRESHOLD = 59

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()

# GET RAW EPIDEMIOLOGY TABLE
source = "WRD_ECDC"
exclude = ['Other continent', 'Asia', 'Europe', 'America', 'Africa', 'Oceania']
cols = 'countrycode, country, date, confirmed, dead'
sql_command = """SELECT """ + cols + """ FROM epidemiology WHERE adm_area_1 IS NULL AND source = %(source)s"""
raw_epidemiology = pd.read_sql(sql_command, conn, params={'source':source})
raw_epidemiology = raw_epidemiology.sort_values(by=['countrycode', 'date'])
raw_epidemiology = raw_epidemiology[~raw_epidemiology['country'].isin(exclude)].reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode', 'date']].duplicated().any()

# GET RAW MOBILITY TABLE
source = 'GOOGLE_MOBILITY'
mobilities = ['residential','workplace','transit_stations','retail_recreation']
cols = 'countrycode, country, date, ' +  ', '.join(mobilities)
sql_command = """SELECT """ + cols + """ FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility.sort_values(by=['countrycode', 'date']).reset_index(drop=True)
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
testing = pd.DataFrame(columns=['countrycode','date', 'total_tests', 'new_tests', 'new_tests_smoothed', 'positive_rate'])

for country in tqdm(countries, desc='Pre-processing Testing Data'):
    data = raw_testing_data[raw_testing_data['countrycode'] == country].reset_index(drop=True)
    if len(data['new_tests'].dropna()) == 0:
        continue
    data = data.iloc[
           data[data['new_tests'].notnull()].index[0]:data[data['new_tests'].notnull()].index[-1]].set_index('date')
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
    'new_tests_smoothed': np.empty(0),
    'positive_rate': np.empty(0),
    'days_since_t0_pop':np.empty(0),
    'days_since_t0_1_dead':np.empty(0),
    'days_since_t0_5_dead':np.empty(0),
    'days_since_t0_10_dead':np.empty(0)
}

if SAVE_PLOTS:
    for i in range(0,6):
        os.makedirs(PLOT_PATH + 'epidemiological/class_' + str(i) + '/', exist_ok=True)


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
    new_tests_smoothed = np.repeat(np.nan, len(data))
    positive_rate = np.repeat(np.nan, len(data))

    x = np.arange(len(data['date']))
    y = data['new_per_day'].values
    ys = csaps(x, y, x, smooth=SMOOTH)
    z = data['dead_per_day'].values
    zs = csaps(x, z, x, smooth=SMOOTH)

    if len(testing_data) > 1:
        tests = data[['date']].merge(
            testing_data[['date','total_tests']],how='left',on='date')['total_tests'].values
        new_tests = data[['date']].merge(
            testing_data[['date','new_tests']],how='left',on='date')['new_tests'].values

        x_testing = np.arange(len(new_tests[~np.isnan(new_tests)]))
        y_testing = new_tests[~np.isnan(new_tests)]
        ys_testing = csaps(x_testing, y_testing, x_testing, smooth=SMOOTH)

        new_tests_smoothed[np.where(~np.isnan(new_tests))] = ys_testing
        positive_rate[np.where(~np.isnan(new_tests))] = ys[np.where(~np.isnan(new_tests))] / ys_testing
        positive_rate[positive_rate > 1] = np.nan

    population = np.nan if len(wb_statistics[wb_statistics['countrycode']==country]['value'])==0 else \
        wb_statistics[wb_statistics['countrycode']==country]['value'].iloc[0]
    t0 = np.nan if len(data[data['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date']) == 0 else \
        data[data['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date'].iloc[0]
    t0_relative = np.nan if len(data[((data['confirmed']/population)*1000000) >= POP_RELATIVE_T0_THRESHOLD]) == 0 else \
        data[((data['confirmed']/population)*1000000) >= POP_RELATIVE_T0_THRESHOLD]['date'].iloc[0]
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
    epidemiology_series['new_tests_smoothed'] = np.concatenate(
        (epidemiology_series['new_tests_smoothed'], new_tests_smoothed))
    epidemiology_series['positive_rate'] = np.concatenate(
        (epidemiology_series['positive_rate'], positive_rate))
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
        x = np.arange(len(data['date']))
        y = data[mobility_type].values
        ys = csaps(x, y, x, smooth=SMOOTH)

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
    data['tests_class'] = np.nan
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
    country_series = epidemiology_series[epidemiology_series['countrycode'] == country]
    data['country'] = country_series['country'].iloc[0]
    data['population'] = np.nan if len(wb_statistics[wb_statistics['countrycode'] == country]) == 0 else \
        wb_statistics[wb_statistics['countrycode'] == country]['value'].values[0]
    data['t0'] = np.nan if len(country_series[country_series['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date']) == 0 else \
        country_series[country_series['confirmed']>=ABSOLUTE_T0_THRESHOLD]['date'].iloc[0]
    data['t0_relative'] = np.nan if len(country_series[(country_series['confirmed'] / data['population'] * 1000000
                                      >= POP_RELATIVE_T0_THRESHOLD)]['date']) == 0 else \
        country_series[(country_series['confirmed'] / data['population'] * 1000000
                                      >= POP_RELATIVE_T0_THRESHOLD)]['date'].iloc[0]
    data['t0_1_dead'] = np.nan if len(country_series[country_series['dead']>=1]['date']) == 0 else \
        country_series[country_series['dead']>=1]['date'].iloc[0]
    data['t0_5_dead'] = np.nan if len(country_series[country_series['dead']>=5]['date']) == 0 else \
        country_series[country_series['dead']>=5]['date'].iloc[0]
    data['t0_10_dead'] = np.nan if len(country_series[country_series['dead']>=10]['date']) == 0 else \
        country_series[country_series['dead']>=10]['date'].iloc[0]

    cases_prominence_threshold = max(PROMINENCE_THRESHOLD, 
                                     RELATIVE_PROMINENCE_THRESHOLD*np.nanmax(country_series['new_per_day_smooth']))
    peak_characteristics = find_peaks(country_series['new_per_day_smooth'].values,
                                      prominence=cases_prominence_threshold, distance=DISTANCE)
    genuine_peaks = peak_characteristics[0]
    # Label country wave status
    data['class'] = 2*len(genuine_peaks)
    # Increase the class by 1 if the country is entering a new peak
    if data['class'] > 0 and genuine_peaks[-1]<len(country_series):
        # Entering a new peak iff: after the minima following the last genuine peak, there is a value >=f% of the smallest peak value
        last_peak_date = country_series['date'].values[genuine_peaks[-1]]
        trough_value = min(country_series.loc[country_series["date"]>last_peak_date,"new_per_day_smooth"])
        trough_date = country_series.loc[country_series["new_per_day_smooth"]==trough_value,"date"].values[0]
        max_after_trough = np.nanmax(country_series.loc[country_series["date"]>=trough_date,"new_per_day_smooth"])
        if max_after_trough-trough_value >= RELATIVE_PROMINENCE_THRESHOLD*np.nanmax(country_series['new_per_day_smooth']):
            data['class'] = data['class'] + 1
    elif data['class'] == 0:
        if np.nanmax(country_series["new_per_day_smooth"]) >= CLASS_1_THRESHOLD:
            data['class'] = data['class'] + 1
    
    if len(genuine_peaks) >= 1:
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
    if len(genuine_peaks) >= 2:
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
    dead_prominence_threshold = max(PROMINENCE_THRESHOLD_DEAD, 
                                    RELATIVE_PROMINENCE_THRESHOLD*np.nanmax(country_series['dead_per_day_smooth']))
    dead_peak_characteristics = find_peaks(country_series['dead_per_day_smooth'].values,
                                           prominence=dead_prominence_threshold, distance=DISTANCE)
    genuine_peaks = dead_peak_characteristics[0]
    data['dead_class'] = 2*len(genuine_peaks)
    if data['dead_class'] > 0 and genuine_peaks[-1]<len(country_series):
        last_peak_date = country_series['date'].values[genuine_peaks[-1]]
        trough_value = min(country_series.loc[country_series["date"]>last_peak_date,"dead_per_day_smooth"])
        trough_date = country_series.loc[country_series["dead_per_day_smooth"]==trough_value,"date"].values[0]
        max_after_trough = np.nanmax(country_series.loc[country_series["date"]>=trough_date,"dead_per_day_smooth"])
        if max_after_trough-trough_value >= RELATIVE_PROMINENCE_THRESHOLD*np.nanmax(country_series['dead_per_day_smooth']):
            data['dead_class'] = data['dead_class'] + 1
    elif data['dead_class'] == 0:
        if np.nanmax(country_series["dead_per_day_smooth"]) >= CLASS_1_THRESHOLD_DEAD:
            data['dead_class'] = data['dead_class'] + 1
    
    if len(genuine_peaks) >= 1:
        data['date_dead_peak_1'] = country_series['date'].values[genuine_peaks[0]]
        bases = np.append(dead_peak_characteristics[1]['left_bases'],dead_peak_characteristics[1]['right_bases'])
        data['dead_first_wave_start'] = min(data['t0_1_dead'],country_series['date'].values[max([b for b in bases if b<genuine_peaks[0]])]) \
                                        if not pd.isnull(data['t0_1_dead']) \
                                        else country_series['date'].values[max([b for b in bases if b<genuine_peaks[0]])]
        data['dead_first_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[0]])]
    if len(genuine_peaks) >= 2:
        data['date_dead_peak_2'] = country_series['date'].values[genuine_peaks[1]]
        data['dead_second_wave_start'] = country_series['date'].values[max([b for b in bases if b<genuine_peaks[1]])]
        data['dead_second_wave_end'] = country_series['date'].values[min([b for b in bases if b>genuine_peaks[1]])]
    # Classify wave status for tests
    if data['testing_available']:
        tests_prominence_threshold = max(PROMINENCE_THRESHOLD_TESTS, 
                                        RELATIVE_PROMINENCE_THRESHOLD*np.nanmax(country_series['new_tests_smoothed']))
        tests_peak_characteristics = find_peaks(country_series['new_tests_smoothed'].values,
                                                prominence=tests_prominence_threshold, distance=DISTANCE)
        genuine_peaks = tests_peak_characteristics[0]
        data['tests_class'] = 2*len(genuine_peaks)
        if data['tests_class'] > 0 and genuine_peaks[-1]<len(country_series):
            last_peak_date = country_series['date'].values[genuine_peaks[-1]]
            trough_value = min(country_series.loc[country_series["date"]>last_peak_date,"new_tests_smoothed"])
            trough_date = country_series.loc[country_series["new_tests_smoothed"]==trough_value,"date"].values[0]
            max_after_trough = np.nanmax(country_series.loc[country_series["date"]>=trough_date,"new_tests_smoothed"])
            if max_after_trough-trough_value >= RELATIVE_PROMINENCE_THRESHOLD*np.nanmax(country_series['new_tests_smoothed']):
                data['tests_class'] = data['tests_class'] + 1
        elif data['tests_class'] == 0:
            if np.nanmax(country_series["new_tests_smoothed"]) >= CLASS_1_THRESHOLD_TESTS:
                data['tests_class'] = data['tests_class'] + 1
            
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
                 label='New Cases per Day Spline')
        axes[0].plot([country_series['date'].values[i]
                  for i in peak_characteristics[0]],
                 [country_series['new_per_day_smooth'].values[i]
                  for i in peak_characteristics[0]], "X", ms=20, color='red')
        if data['class']>=2:
            axes[0].plot(data['first_wave_start'],country_series.loc[country_series['date']==data['first_wave_start'],'new_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[0].text(data['first_wave_start'],country_series.loc[country_series['date']==data['first_wave_start'],'new_per_day_smooth'].values[0], 'First Wave Start')
            axes[0].plot(data['first_wave_end'],country_series.loc[country_series['date']==data['first_wave_end'],'new_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[0].text(data['first_wave_end'],country_series.loc[country_series['date']==data['first_wave_end'],'new_per_day_smooth'].values[0], 'First Wave End')
        if data['class']>=4:
            axes[0].plot(data['second_wave_start'],country_series.loc[country_series['date']==data['second_wave_start'],'new_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[0].text(data['second_wave_start'],country_series.loc[country_series['date']==data['second_wave_start'],'new_per_day_smooth'].values[0], 'Second Wave Start')
            axes[0].plot(data['second_wave_end'],country_series.loc[country_series['date']==data['second_wave_end'],'new_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[0].text(data['second_wave_end'],country_series.loc[country_series['date']==data['second_wave_end'],'new_per_day_smooth'].values[0], 'Second Wave End')
        axes[1].plot(country_series['date'].values,
                 country_series['dead_per_day'].values,
                 label='Deaths per Day')
        axes[1].plot(country_series['date'].values,
                 country_series['dead_per_day_smooth'].values,
                 label='Deaths per Day Spline')
        axes[1].plot([country_series['date'].values[i]
                  for i in dead_peak_characteristics[0]],
                 [country_series['dead_per_day_smooth'].values[i]
                  for i in dead_peak_characteristics[0]], "X", ms=20, color='red')
        if data['dead_class']>=2:
            axes[1].plot(data['dead_first_wave_start'],country_series.loc[country_series['date']==data['dead_first_wave_start'],'dead_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[1].text(data['dead_first_wave_start'],country_series.loc[country_series['date']==data['dead_first_wave_start'],'dead_per_day_smooth'].values[0], 'Deaths First Wave Start')
            axes[1].plot(data['dead_first_wave_end'],country_series.loc[country_series['date']==data['dead_first_wave_end'],'dead_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[1].text(data['dead_first_wave_end'],country_series.loc[country_series['date']==data['dead_first_wave_end'],'dead_per_day_smooth'].values[0], 'Deaths First Wave End')
        if data['dead_class']>=4:
            axes[1].plot(data['dead_second_wave_start'],country_series.loc[country_series['date']==data['dead_second_wave_start'],'dead_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[1].text(data['dead_second_wave_start'],country_series.loc[country_series['date']==data['dead_second_wave_start'],'dead_per_day_smooth'].values[0], 'Deaths Second Wave Start')
            axes[1].plot(data['dead_second_wave_end'],country_series.loc[country_series['date']==data['dead_second_wave_end'],'dead_per_day_smooth'].values[0],
                         "X", ms=20, color='orange')
            axes[1].text(data['dead_second_wave_end'],country_series.loc[country_series['date']==data['dead_second_wave_end'],'dead_per_day_smooth'].values[0], 'Deaths Second Wave End')
        if data['testing_available']:
            axes[2].plot(country_series['date'].values,
                     country_series['new_tests'].values,
                     label='Tests per Day')
            axes[2].plot(country_series['date'].values,
                     country_series['new_tests_smoothed'].values,
                     label='Tests per Day Spline')
            axes[2].plot([country_series['date'].values[i]
                      for i in tests_peak_characteristics[0]],
                     [country_series['new_tests_smoothed'].values[i]
                      for i in tests_peak_characteristics[0]], "X", ms=20, color='red')
        axes[0].set_title('New Cases per Day')
        axes[0].set_ylabel('New Cases per Day')
        axes[1].set_title('Deaths per Day')
        axes[1].set_ylabel('Deaths per Day')
        axes[2].set_title('Tests per Day')
        axes[2].set_ylabel('Tests per Day')
        f.suptitle('Cases, Deaths and Tests per Day for ' + country)
        plt.savefig(PLOT_PATH + 'epidemiological/class_' +str(data['class']) +'/' + country + '.png')
        plt.close('all')    
    
    epidemiology_panel = epidemiology_panel.append(data,ignore_index=True)

if SAVE_CSV:
    epidemiology_panel.to_csv(CSV_PATH + 'epidemiology_panel.csv')


mobility_panel = pd.DataFrame(columns=['countrycode','country'] +
                                      [mobility_type + '_max' for mobility_type in mobilities] +
                                      [mobility_type + '_max_date' for mobility_type in mobilities] +
                                      [mobility_type + '_quarantine_fatigue' for mobility_type in mobilities])

countries = mobility['countrycode'].unique()
for country in tqdm(countries, desc='Processing Mobility Panel Data'):
    data = dict()
    data['countrycode'] = country
    data['country'] = mobility_series[mobility_series['countrycode']==country]['country'].iloc[0]
    for mobility_type in mobilities:
        data[mobility_type + '_max'] = mobility_series[
            mobility_series['countrycode']==country][mobility_type + '_smooth'].max()
        data[mobility_type + '_max_date'] = mobility_series[
            mobility_series['countrycode'] == country].iloc[
            mobility_series[mobility_series['countrycode'] == country][mobility_type + '_smooth'].argmax()]['date']
        data[mobility_type + '_quarantine_fatigue'] = np.nan

        mob_data_to_fit = mobility_series[
                        mobility_series['countrycode']==country][mobility_type + '_smooth'].iloc[
                        mobility_series[mobility_series['countrycode']==country][mobility_type + '_smooth'].argmax()::]
        mob_data_to_fit = mob_data_to_fit.dropna()
        if len(mob_data_to_fit) != 0:
            data[mobility_type + '_quarantine_fatigue'] = LinearRegression().fit(
                np.arange(len(mob_data_to_fit)).reshape(-1,1),mob_data_to_fit.values).coef_[0]

    mobility_panel = mobility_panel.append(data, ignore_index=True)
    continue

    
government_response_panel = pd.DataFrame(columns=['countrycode', 'country', 'max_si','date_max_si',
                                                  'si_days_to_max_si', 'si_at_t0','si_at_peak_1',
                                                  'si_days_to_threshold',
                                                  'si_days_above_threshold','si_days_above_threshold_first_wave'] +
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
"""
DEPRECATED FIGURES
'''
PART 3 - SAVING FIGURE 1a
Chloropleth of days to t0 per country
'''
start_date = epidemiology_series['date'].min()
figure_1a = pd.DataFrame(columns=['countrycode','country','days_to_t0','days_to_t0_pop'])
figure_1a['countrycode'] = epidemiology_panel['countrycode'].values
figure_1a['country'] = epidemiology_panel['country'].values
figure_1a['days_to_t0'] = (epidemiology_panel['t0']-start_date).apply(lambda x: x.days)
figure_1a['days_to_t0_pop'] = (epidemiology_panel['t0_relative']-start_date).apply(lambda x: x.days)
# It looks like pandas cannot correctly serialise the geometry column so this
# is being commented out. Possibly this merge could be done at plot time, or
# the data could be pickled instead of written to a flat CSV. Since it is just
# a left merge with the country code this should be feasible.
figure_1a = figure_1a.merge(map_data[['countrycode']],on='countrycode',how = 'left').dropna()
figure_1a = figure_1a.merge(map_data[['countrycode','geometry']],on='countrycode',how = 'left').dropna()
if SAVE_CSV:
    # Asserts have been added because there was a wierd bug encountered in
    # saving figure_1a to CSV and they were introduced to help track it down.
    assert isinstance(figure_1a, pd.core.frame.DataFrame), "figure_1a is not a pandas dataframe..."
    assert hasattr(figure_1a, "to_csv"), "figure_1a does not have a to_csv method..."
    figure_1a.to_csv(CSV_PATH + 'figure_1a.csv', sep=';')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 3 - SAVING FIGURE 1b
Chloropleth of wave status per country
'''
figure_1b = epidemiology_panel[['countrycode','country','class']].dropna()
# It looks like pandas cannot correctly serialise the geometry column so this
# is being commented out. Possibly this merge could be done at plot time, or
# the data could be pickled instead of written to a flat CSV. Since it is just
# a left merge with the country code this should be feasible.
figure_1b = figure_1b.merge(map_data[['countrycode']],on='countrycode',how = 'left').dropna()
figure_1b = figure_1b.merge(map_data[['countrycode','geometry']],on='countrycode',how = 'left').dropna()
if SAVE_CSV:
    figure_1b.to_csv(CSV_PATH + 'figure_1b.csv',sep=';')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 4 - SAVING FIGURE 2a
Time series of stringency index
'''
data = epidemiology_series[['countrycode','country','date','days_since_t0','days_since_t0_pop']].merge(
    epidemiology_panel[['countrycode','class']], on='countrycode',how='left').merge(
    government_response_series[['countrycode','date','si']],on=['countrycode','date'],how='left').dropna()
figure_2a = pd.DataFrame(columns=['COUNTRYCODE','COUNTRY','CLASS','t','stringency_index'])
figure_2a['COUNTRYCODE'] = data['countrycode']
figure_2a['COUNTRY'] = data['country']
figure_2a['CLASS'] = data['class']
figure_2a['t'] = data['days_since_t0']
figure_2a['t_pop'] = data['days_since_t0_pop']
figure_2a['stringency_index'] = data['si']
if SAVE_CSV:
    figure_2a.to_csv(CSV_PATH + 'figure_2a.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 5 - SAVING FIGURE 2b
Time series of mobility
'''
data = epidemiology_series[['countrycode','country','date','days_since_t0','days_since_t0_pop']].merge(
    epidemiology_panel[['countrycode','class']], on='countrycode',how='left').merge(
    mobility_series[['countrycode','date','residential_smooth']],on=['countrycode','date'],how='left').dropna()
figure_2b = pd.DataFrame(columns=['COUNTRYCODE','COUNTRY','CLASS','t','residential_smooth'])
figure_2b['COUNTRYCODE'] = data['countrycode']
figure_2b['COUNTRY'] = data['country']
figure_2b['CLASS'] = data['class']
figure_2b['t'] = data['days_since_t0']
figure_2b['t_pop'] = data['days_since_t0_pop']
figure_2b['residential_smooth'] = data['residential_smooth']
if SAVE_CSV:
    figure_2b.to_csv(CSV_PATH + 'figure_2b.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 6 - SAVING FIGURE 2c
Scatterplot of response time against cases
'''
class_coarse = {
    0:np.nan,
    1:'EPI_FIRST_WAVE',
    2:'EPI_FIRST_WAVE',
    3:'EPI_SECOND_WAVE',
    4:'EPI_SECOND_WAVE'
}
data = epidemiology_panel[['countrycode', 'country', 'class' , 'population', 'last_confirmed']]
data['class_coarse'] = data['class'].apply(lambda x:class_coarse[x])
data['last_confirmed_per_10k'] = 10000 * epidemiology_panel['last_confirmed'] / epidemiology_panel['population']
data['class_coarse'] = data['class'].apply(lambda x: class_coarse[x])
data = data.merge(government_response_panel[['countrycode', 'response_time','response_time_pop']],
                  how='left', on='countrycode').dropna()
figure_2c = pd.DataFrame(columns=['COUNTRYCODE', 'COUNTRY', 'GOV_MAX_SI_DAYS_FROM_T0',
                                 'CLASS_COARSE', 'POPULATION', 'EPI_CONFIRMED', 'EPI_CONFIRMED_PER_10K'])
figure_2c['COUNTRYCODE'] = data['countrycode']
figure_2c['COUNTRY'] = data['country']
figure_2c['GOV_MAX_SI_DAYS_FROM_T0'] = data['response_time']
figure_2c['GOV_MAX_SI_DAYS_FROM_T0_POP'] = data['response_time_pop']
figure_2c['CLASS'] = data['class']
figure_2c['CLASS_COARSE'] = data['class_coarse']
figure_2c['POPULATION'] = data['population']
figure_2c['EPI_CONFIRMED'] = data['last_confirmed']
figure_2c['EPI_CONFIRMED_PER_10K'] = data['last_confirmed_per_10k']
if SAVE_CSV:
    figure_2c.to_csv(CSV_PATH + 'figure_2c.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 7 - SAVING FIGURE 3
3x3 time series subplots of cases, deaths and testing
'''
figure_3 = epidemiology_series[[
    'country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
    'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smoothed', 'positive_rate']]
if SAVE_CSV:
    figure_3.to_csv(CSV_PATH + 'figure_3.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 8 - SAVING FIGURE 4
Time series of cases, stringency and mobility for second wave countries
'''
# noinspection PyRedeclaration
class_coarse = {
    0:'EPI_OTHER',
    1:'EPI_FIRST_WAVE',
    2:'EPI_FIRST_WAVE',
    3:'EPI_SECOND_WAVE',
    4:'EPI_SECOND_WAVE'
}
data = epidemiology_series[['countrycode','country','date','confirmed','new_cases_per_10k','days_since_t0',
                            'days_since_t0_pop']].merge(
    government_response_series[['countrycode','date','si']],on=['countrycode','date'],how='inner').merge(
    mobility_series[['countrycode','date','residential_smooth']],on=['countrycode','date'],how='inner').merge(
    epidemiology_panel[['countrycode','class','t0','t0_relative']],on=['countrycode'],how='left').merge(
    government_response_panel[['countrycode','c6_stay_at_home_requirements_raised',
                               'c6_stay_at_home_requirements_lowered',
                               'c6_stay_at_home_requirements_raised_again']],on='countrycode',how='left')
data['class_coarse'] = data['class'].apply(lambda x:class_coarse[x])
figure_4 = pd.DataFrame(columns=['COUNTRYCODE', 'COUNTRY', 'T0', 'T0_POP', 'date', 'stringency_index', 'CLASS',
                                 'CLASS_COARSE', 'GOV_C6_RAISED_DATE', 'GOV_C6_LOWERED_DATE','GOV_C6_RAISED_AGAIN_DATE',
                                 'residential_smooth', 't', 'confirmed', 'new_per_day_smooth_per10k'])
figure_4['COUNTRYCODE'] = data['countrycode']
figure_4['COUNTRY'] = data['country']
figure_4['T0'] = data['t0']
figure_4['T0_POP'] = data['t0_relative']
figure_4['date'] = data['date']
figure_4['stringency_index'] = data['si']
figure_4['CLASS'] = data['class']
figure_4['CLASS_COARSE'] = data['class_coarse']
figure_4['GOV_C6_RAISED_DATE'] = data['c6_stay_at_home_requirements_raised']
figure_4['GOV_C6_LOWERED_DATE'] = data['c6_stay_at_home_requirements_lowered']
figure_4['GOV_C6_RAISED_AGAIN_DATE'] = data['c6_stay_at_home_requirements_raised_again']
figure_4['residential_smooth'] = data['residential_smooth']
figure_4['t'] = data['days_since_t0']
figure_4['t_pop'] = data['days_since_t0_pop']
figure_4['confirmed'] = data['confirmed']
figure_4['new_per_day_smooth_per10k'] = data['new_cases_per_10k']
if SAVE_CSV:
    figure_4.to_csv(CSV_PATH + 'figure_4.csv')
"""
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
    figure_1.to_csv(CSV_PATH + 'figure_1.csv', sep=';')
    
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 4 - FIGURE 2
'''

figure_2 = epidemiology_series[[
    'country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
    'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smoothed', 'positive_rate']]

if SAVE_CSV:
    figure_2.to_csv(CSV_PATH + 'figure_2.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 5 - FIGURE 3
'''

data = epidemiology_series[['countrycode','country','date', 'days_since_t0_pop', 'days_since_t0_1_dead', 'days_since_t0_5_dead', 'days_since_t0_10_dead']].merge(
    epidemiology_panel[['countrycode','class']], on='countrycode',how='left').merge(
    government_response_series[['countrycode','date','si']],on=['countrycode','date'],how='left').dropna()

figure_3a = pd.DataFrame(columns=['COUNTRYCODE','COUNTRY','CLASS','t_pop','t_1_dead','t_5_dead','t_10_dead''stringency_index'])
figure_3a['COUNTRYCODE'] = data['countrycode']
figure_3a['COUNTRY'] = data['country']
figure_3a['CLASS'] = data['class']
figure_3a['t_pop'] = data['days_since_t0_pop']
figure_3a['t_1_dead'] = data['days_since_t0_1_dead']
figure_3a['t_5_dead'] = data['days_since_t0_5_dead']
figure_3a['t_10_dead'] = data['days_since_t0_10_dead']
figure_3a['stringency_index'] = data['si']

if SAVE_CSV:
    figure_3a.to_csv(CSV_PATH + 'figure_3a.csv')

data = epidemiology_series[['countrycode','country','date','days_since_t0_pop', 'days_since_t0_1_dead', 'days_since_t0_5_dead', 'days_since_t0_10_dead']].merge(
    epidemiology_panel[['countrycode','class']], on='countrycode',how='left').merge(
    mobility_series[['countrycode','date','residential','residential_smooth','workplace','workplace_smooth',
                     'transit_stations','transit_stations_smooth','retail_recreation','retail_recreation_smooth']].dropna(how='all'),on=['countrycode','date'],how='inner')

figure_3b = pd.DataFrame(columns=['COUNTRYCODE','COUNTRY','CLASS','t','residential','residential_smooth'])
figure_3b['COUNTRYCODE'] = data['countrycode']
figure_3b['COUNTRY'] = data['country']
figure_3b['CLASS'] = data['class']
figure_3b['t_pop'] = data['days_since_t0_pop']
figure_3b['t_1_dead'] = data['days_since_t0_1_dead']
figure_3b['t_5_dead'] = data['days_since_t0_5_dead']
figure_3b['t_10_dead'] = data['days_since_t0_10_dead']
figure_3b['residential'] = data['residential']
figure_3b['residential_smooth'] = data['residential_smooth']
figure_3b['workplace'] = data['workplace']
figure_3b['workplace_smooth'] = data['workplace_smooth']
figure_3b['transit_stations'] = data['transit_stations']
figure_3b['transit_stations_smooth'] = data['transit_stations_smooth']
figure_3b['retail_recreation'] = data['retail_recreation']
figure_3b['retail_recreation_smooth'] = data['retail_recreation_smooth']

if SAVE_CSV:
    figure_3b.to_csv(CSV_PATH + 'figure_3b.csv')

class_coarse = {
    0:np.nan,
    1:'EPI_FIRST_WAVE',
    2:'EPI_FIRST_WAVE',
    3:'EPI_SECOND_WAVE',
    4:'EPI_SECOND_WAVE',
    5:'EPI_THIRD_WAVE',
    6:'EPI_THIRD_WAVE'
}

data = epidemiology_panel[['countrycode', 'country', 'class' , 'population', 'last_confirmed','last_dead',
                           't0_relative','t0_1_dead','t0_5_dead','t0_10_dead',
                           'first_wave_start','first_wave_end','duration_first_wave',
                           'dead_first_wave', 'dead_second_wave',
                           'dead_peak_1','dead_peak_2']]
data['class_coarse'] = data['class'].apply(lambda x:class_coarse[x])
data['last_confirmed_per_10k'] = 10000 * epidemiology_panel['last_confirmed'] / epidemiology_panel['population']
data['last_dead_per_10k'] = 10000 * epidemiology_panel['last_dead'] / epidemiology_panel['population']
data = data.merge(government_response_panel[['countrycode','si_days_to_max_si','max_si'] +
                                            [flag + '_at_t0' for flag in flags + ['si']] +
                                            [flag + '_at_peak_1' for flag in flags + ['si']] +
                                            [flag + '_days_to_threshold' for flag in flags + ['si']] +
                                            [flag + '_days_above_threshold_first_wave' for flag in flags + ['si']]],
                  how='left', on='countrycode')

figure_3c = pd.DataFrame(columns=['COUNTRYCODE', 'COUNTRY', 'GOV_MAX_SI_DAYS_FROM_T0',
                                 'CLASS_COARSE', 'POPULATION', 'EPI_CONFIRMED', 'EPI_CONFIRMED_PER_10K',
                                 'EPI_DEAD', 'EPI_DEAD_PER_10K',
                                 'T0_POP','T0_1_DEAD','T0_5_DEAD','T0_10_DEAD',
                                 'EPI_DURATION_FIRST_WAVE','MAX_SI',
                                 'DEAD_FIRST_WAVE', 'DEAD_SECOND_WAVE',
                                 'DEAD_PEAK_1','DEAD_PEAK_2'] +
                                 [flag.upper() + '_AT_T0' for flag in flags + ['si']] +
                                 [flag.upper() + '_AT_PEAK_1' for flag in flags + ['si']] +
                                 [flag.upper() + '_DAYS_TO_THRESHOLD' for flag in flags + ['si']] +
                                 [flag.upper() + '_DAYS_ABOVE_THRESHOLD_FIRST_WAVE' for flag in flags + ['si']])

figure_3c['COUNTRYCODE'] = data['countrycode']
figure_3c['COUNTRY'] = data['country']
figure_3c['GOV_MAX_SI_DAYS_FROM_T0'] = data['si_days_to_max_si']
figure_3c['CLASS'] = data['class']
figure_3c['CLASS_COARSE'] = data['class_coarse']
figure_3c['POPULATION'] = data['population']
figure_3c['EPI_CONFIRMED'] = data['last_confirmed']
figure_3c['EPI_CONFIRMED_PER_10K'] = data['last_confirmed_per_10k']
figure_3c['EPI_DEAD'] = data['last_dead']
figure_3c['EPI_DEAD_PER_10K'] = data['last_dead_per_10k']
figure_3c['T0_POP'] = data['t0_relative']
figure_3c['T0_1_DEAD'] = data['t0_1_dead']
figure_3c['T0_5_DEAD'] = data['t0_5_dead']
figure_3c['T0_10_DEAD'] = data['t0_10_dead']
figure_3c['EPI_DURATION_FIRST_WAVE'] = data['duration_first_wave']
figure_3c['MAX_SI'] = data['max_si']
figure_3c['DEAD_FIRST_WAVE'] = data['dead_first_wave']
figure_3c['DEAD_SECOND_WAVE'] = data['dead_second_wave']
figure_3c['DEAD_PEAK_1'] = data['dead_peak_1']
figure_3c['DEAD_PEAK_2'] = data['dead_peak_2']
for flag in flags + ['si']:
    figure_3c[flag.upper() + '_AT_T0'] = data[flag + '_at_t0']
    figure_3c[flag.upper() + '_AT_PEAK_1'] = data[flag + '_at_peak_1']
    figure_3c[flag.upper() + '_DAYS_TO_THRESHOLD'] = data[flag + '_days_to_threshold']
    figure_3c[flag.upper() + '_DAYS_ABOVE_THRESHOLD_FIRST_WAVE'] = data[flag + '_days_above_threshold_first_wave']

if SAVE_CSV:
    figure_3c.to_csv(CSV_PATH + 'figure_3c.csv')

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
    figure_4.to_csv(CSV_PATH + 'figure_4.csv', sep=';')
    
# -------------------------------------------------------------------------------------------------------------------- #
  
'''
PART 6.5 - FIGURE 4A STACKED AREA PLOT BY STATE
'''    
# GET RAW CONFIRMED CASES TABLE FOR USA STATES
cols = 'countrycode, adm_area_1, date, confirmed'
sql_command = """SELECT """ + cols + """ FROM epidemiology WHERE countrycode = 'USA' AND source = 'USA_NYT' AND adm_area_1 IS NOT NULL AND adm_area_2 IS NULL"""
raw_usa = pd.read_sql(sql_command, conn)
raw_usa = raw_usa.sort_values(by=['adm_area_1', 'date']).reset_index(drop=True)

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
    x = np.arange(len(data['date']))
    y = data['new_per_day'].values
    ys = csaps(x, y, x, smooth=SMOOTH)
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
        'countrycode', 'c6_stay_at_home_requirements_total_days', 'response_time',
        'c6_stay_at_home_requirements_response_time', 'date_max_si']], on=['countrycode'], how='left').merge(
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
table1['days_with_stay_at_home'] = data['c6_stay_at_home_requirements_total_days']
table1['response_time_si'] = data['response_time']
table1['response_time_stay_at_home'] = data['c6_stay_at_home_requirements_response_time']
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
    np.savetxt(CSV_PATH + 'last_updated.txt', [datetime.datetime.today().date().strftime('%Y-%m-%d')], fmt='%s')
# -------------------------------------------------------------------------------------------------------------------- #
