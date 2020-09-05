import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import os
import warnings
from tqdm import tqdm
import datetime

from csaps import csaps
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

'''
INTITALISE SCRIPT PARAMETERS
'''

SAVE_PLOTS = False
SAVE_CSV = True
PLOT_PATH = './plots/'
CSV_PATH = './data/'
SMOOTH = 0.001
DISTANCE = 21
PROMINENCE_THRESHOLD = 5
ABSOLUTE_T0_THRESHOLD = 1000
POP_RELATIVE_T0_THRESHOLD = 5 #per million people
TEST_LAG = 0 #Lag between test date and test results

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

sql_command = """SELECT * FROM epidemiology WHERE source = %(source)s"""
raw_epidemiology = pd.read_sql(sql_command, conn, params={'source': source})
raw_epidemiology = raw_epidemiology[raw_epidemiology['adm_area_1'].isnull()].sort_values(by=['countrycode', 'date'])
raw_epidemiology = raw_epidemiology[~raw_epidemiology['country'].isin(exclude)].reset_index(drop=True)
raw_epidemiology = raw_epidemiology[['countrycode', 'country', 'date', 'confirmed', 'dead']]
# Check no conflicting values for each country and date
assert not raw_epidemiology[['countrycode', 'date']].duplicated().any()

# GET RAW MOBILITY TABLE
source = 'GOOGLE_MOBILITY'
mobilities = ['residential']

sql_command = """SELECT * FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility[['countrycode', 'country', 'date'] + mobilities]
raw_mobility = raw_mobility.sort_values(by=['countrycode', 'date']).reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_mobility[['countrycode', 'date']].duplicated().any()

# GET RAW GOVERNMENT RESPONSE TABLE
flags = ['c6_stay_at_home_requirements']
flag_thresholds = {'c6_stay_at_home_requirements': 2}

sql_command = """SELECT * FROM government_response"""
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

# GET COUNTRY POPULATIONS (2011 - 2019 est.)
indicator_code = 'SP.POP.TOTL'
sql_command = """SELECT countrycode, value, year FROM world_bank WHERE 
adm_area_1 IS NULL AND indicator_code = %(indicator_code)s"""
wb_statistics = pd.read_sql(sql_command, conn, params={'indicator_code': indicator_code})
assert len(wb_statistics) == len(wb_statistics['countrycode'].unique())
wb_statistics = wb_statistics.sort_values(by=['countrycode', 'year'], ascending=[True, False]).reset_index(drop=True)

# OWID DATA
raw_testing_data = pd.read_csv('./owid-covid-data.csv', parse_dates=['date'])[[
    'iso_code', 'date', 'new_tests', 'new_tests_smoothed', 'positive_rate']].rename(columns={'iso_code':'countrycode'})
raw_testing_data['date'] = raw_testing_data['date'].apply(lambda x:x.date())

# GET COUNTRY LABELS
class_dictionary = {
    'EPI_ENTERING_FIRST': 1,
    'EPI_PAST_FIRST': 2,
    'EPI_ENTERING_SECOND': 3,
    'EPI_PAST_SECOND': 4}
labelled_columns = pd.read_csv('./peak_labels.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 0 - PRE-PROCESSING
'''

# EPIDEMIOLOGY PROCESSING
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode', 'country', 'date', 'confirmed', 'new_per_day','dead_per_day'])
for country in countries:
    data = raw_epidemiology[raw_epidemiology['countrycode'] == country].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
    data[['countrycode', 'country']] = data[['countrycode', 'country']].fillna(method='backfill')
    data['confirmed'] = data['confirmed'].interpolate(method='linear')
    data['new_per_day'] = data['confirmed'].diff()
    data.reset_index(inplace=True)
    data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index)] = \
        data['new_per_day'].iloc[np.array(epidemiology[epidemiology['new_per_day'] < 0].index) - 1]
    data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
    data['dead'] = data['dead'].interpolate(method='linear')
    data['dead_per_day'] = data['dead'].diff()
    data.reset_index(inplace=True)
    data['dead_per_day'].iloc[np.array(data[data['dead_per_day'] < 0].index)] = \
        data['dead_per_day'].iloc[np.array(epidemiology[epidemiology['dead_per_day'] < 0].index) - 1]
    data['dead_per_day'] = data['dead_per_day'].fillna(method='bfill')
    epidemiology = pd.concat((epidemiology, data)).reset_index(drop=True)
    continue

# MOBILITY PROCESSING
countries = raw_mobility['countrycode'].unique()
mobility = pd.DataFrame(columns=['countrycode', 'country', 'date'] + mobilities)

for country in countries:
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

for country in countries:
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
testing = pd.DataFrame(columns=['countrycode','date', 'new_tests', 'new_tests_smoothed', 'positive_rate'])

for country in countries:
    data = raw_testing_data[raw_testing_data['countrycode'] == country].reset_index(drop=True)
    if (len(data['new_tests'].dropna()) == 0):
        continue
    data = data.iloc[
           data[data['new_tests'].notnull()].index[0]:data[data['new_tests'].notnull()].index[-1]].set_index('date')
    data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
    data[['countrycode']] = data[['countrycode']].fillna(method='backfill')
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
    'new_tests': np.empty(0),
    'new_tests_smoothed': np.empty(0),
    'positive_rate': np.empty(0),
    'days_since_t0_pop':np.empty(0)
}

if SAVE_PLOTS:
    os.makedirs(PLOT_PATH + 'epidemiological/', exist_ok=True)

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

    new_tests = np.repeat(np.nan, len(data))
    new_tests_smoothed = np.repeat(np.nan, len(data))
    positive_rate = np.repeat(np.nan, len(data))

    x = np.arange(len(data['date']))
    y = data['new_per_day'].values
    ys = csaps(x, y, x, smooth=SMOOTH)
    z = data['dead_per_day'].values
    zs = csaps(x, z, x, smooth=SMOOTH)

    if len(testing_data) > 1:
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
    t0 = np.nan if len(data[data['confirmed']>ABSOLUTE_T0_THRESHOLD]['date']) == 0 else \
        data[data['confirmed']>ABSOLUTE_T0_THRESHOLD]['date'].iloc[0]
    t0_relative = np.nan if len(data[((data['confirmed']/population)*1000000) > POP_RELATIVE_T0_THRESHOLD]) == 0 else \
        data[((data['confirmed']/population)*1000000) > POP_RELATIVE_T0_THRESHOLD]['date'].iloc[0]

    days_since_t0 = np.repeat(np.nan,len(data)) if pd.isnull(t0) else \
        np.array([(date - t0).days for date in data['date'].values])
    days_since_t0_relative = np.repeat(np.nan,len(data)) if pd.isnull(t0_relative) else \
        np.array([(date - t0_relative).days for date in data['date'].values])

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
    epidemiology_series['new_tests'] = np.concatenate(
        (epidemiology_series['new_tests'], new_tests))
    epidemiology_series['new_tests_smoothed'] = np.concatenate(
        (epidemiology_series['new_tests_smoothed'], new_tests_smoothed))
    epidemiology_series['positive_rate'] = np.concatenate(
        (epidemiology_series['positive_rate'], positive_rate))
    epidemiology_series['days_since_t0_pop'] = np.concatenate(
        (epidemiology_series['days_since_t0_pop'], days_since_t0_relative))


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
PEAK_2 - HEIGHT OF SECOND PEAK
DATE_PEAK_1 - DATE OF FIRST WAVE PEAK √
DATE_PEAK_2 - DATE OF SECOND WAVE PEAK
FIRST_WAVE_START - START OF FIRST WAVE  √
FIRST_WAVE_END - END OF FIRST WAVE √
SECOND_WAVE_START - START OF SECOND WAVE √
SECOND_WAVE_END - END OF SECOND WAVE √
LAST_CONFIRMED - LATEST NUMBER OF CONFIRMED CASES √
TESTING DATA AVAILABLE √

PLOT ALL FOR LABELLING √

GOV:
COUNTRYCODE √
COUNTRY √
MAX SI - VALUE OF MAXIMUM SI √
DATE OF PEAK SI - √
RESPONSE TIME - TIME FROM T0 T0 PEAK SI √
RESPONSE TIME - TIME FROM T0_POP TO PEAK SI 
FLAG_RAISED - DATE FLAG RAISED FOR EACH FLAG IN L66 √
FLAG_LOWERED - DATE FLAG LOWERED FOR EACH FLAG IN L66 √
FLAG_RASIED_AGAIN - DATE FLAG RAISED AGAIN FOR EACH FLAG IN L66 √
'''

epidemiology_panel = pd.DataFrame(columns=['countrycode', 'country', 'class', 'population', 't0', 't0_relative',
                                           'peak_1', 'peak_2', 'date_peak_1', 'date_peak_2', 'first_wave_start',
                                           'first_wave_end', 'second_wave_start', 'second_wave_end','last_confirmed',
                                           'testing_available'])

countries = epidemiology['countrycode'].unique()
for country in tqdm(countries, desc='Processing Epidemiological Panel Data'):
    data = dict()
    data['countrycode'] = country
    data['country'] = epidemiology_series[epidemiology_series['countrycode']==country]['country'].iloc[0]
    data['class'] = 0 if np.sum(labelled_columns[labelled_columns['COUNTRYCODE']==country][[
        'EPI_ENTERING_FIRST', 'EPI_PAST_FIRST', 'EPI_ENTERING_SECOND', 'EPI_PAST_SECOND']].values) == 0 else \
        class_dictionary[labelled_columns[labelled_columns['COUNTRYCODE']==country][[
        'EPI_ENTERING_FIRST', 'EPI_PAST_FIRST', 'EPI_ENTERING_SECOND', 'EPI_PAST_SECOND']].idxmax(axis=1).values[0]]
    data['population'] = np.nan if len(wb_statistics[wb_statistics['countrycode'] == country]) == 0 else \
        wb_statistics[wb_statistics['countrycode'] == country]['value'].values[0]
    data['t0'] = np.nan if len(epidemiology_series[(epidemiology_series['countrycode']==country) &
                                     (epidemiology_series['confirmed']>=ABSOLUTE_T0_THRESHOLD)]['date']) == 0 else \
        epidemiology_series[(epidemiology_series['countrycode']==country) &
                                     (epidemiology_series['confirmed']>=ABSOLUTE_T0_THRESHOLD)]['date'].iloc[0]
    data['t0_relative'] = np.nan if len(epidemiology_series[(epidemiology_series['countrycode']==country) &
                                     ((epidemiology_series['confirmed'] / data['population']) * 1000000
                                      >= POP_RELATIVE_T0_THRESHOLD)]['date']) == 0 else \
        epidemiology_series[(epidemiology_series['countrycode']==country) &
                                     ((epidemiology_series['confirmed'] / data['population']) * 1000000
                                      >= POP_RELATIVE_T0_THRESHOLD)]['date'].iloc[0]

    peak_characteristics = find_peaks(
        epidemiology_series[epidemiology_series['countrycode']==country]['new_per_day_smooth'].values,
        prominence=PROMINENCE_THRESHOLD, distance=DISTANCE)
    genuine_peak_indices = labelled_columns[labelled_columns['COUNTRYCODE']==country][[
        'EPI_PEAK_1_GENUINE', 'EPI_PEAK_2_GENUINE', 'EPI_PEAK_3_GENUINE',
        'EPI_PEAK_4_GENUINE']].values.astype(int)[0][0:len(peak_characteristics[0])]
    genuine_peaks = peak_characteristics[0][np.where(genuine_peak_indices != 0)]

    data['peak_1'] = np.nan
    data['peak_2'] = np.nan
    data['date_peak_1'] = np.nan
    data['date_peak_2'] = np.nan
    data['first_wave_start'] = np.nan
    data['first_wave_end'] = np.nan
    data['second_wave_start'] = np.nan
    data['second_wave_end'] = np.nan

    if len(genuine_peaks) >= 1:
        data['peak_1'] = epidemiology_series[
            epidemiology_series['countrycode'] == country]['new_per_day_smooth'].values[genuine_peaks[0]]
        data['date_peak_1'] = epidemiology_series[
            epidemiology_series['countrycode'] == country]['date'].values[genuine_peaks[0]]
        data['first_wave_start'] = epidemiology_series[
            epidemiology_series['countrycode'] == country]['date'].values[
            peak_characteristics[1]['left_bases'][np.where(genuine_peak_indices != 0)][0]]
        data['first_wave_end'] = epidemiology_series[
            epidemiology_series['countrycode'] == country]['date'].values[
            peak_characteristics[1]['right_bases'][np.where(genuine_peak_indices != 0)][0]]

    if len(genuine_peaks) >= 2:
        data['peak_2'] = epidemiology_series[
    epidemiology_series['countrycode'] == country]['new_per_day_smooth'].values[genuine_peaks[1]]
        data['date_peak_2'] = epidemiology_series[
    epidemiology_series['countrycode'] == country]['date'].values[genuine_peaks[1]]
        data['second_wave_start'] = epidemiology_series[
            epidemiology_series['countrycode'] == country]['date'].values[
            peak_characteristics[1]['right_bases'][np.where(genuine_peak_indices != 0)][0]]
        data['second_wave_end'] = epidemiology_series[
    epidemiology_series['countrycode'] == country]['date'].iloc[-1]

    data['last_confirmed'] = epidemiology_series[epidemiology_series['countrycode']==country]['confirmed'].iloc[-1]
    data['testing_available'] = True if len(
        epidemiology_series[epidemiology_series['countrycode']==country]['new_tests'].dropna()) > 0 else False
    epidemiology_panel = epidemiology_panel.append(data,ignore_index=True)

    if SAVE_PLOTS:
        plt.figure(figsize=(20, 7))
        plt.title('New Cases Per Day with Spline Fit for ' + country)
        plt.ylabel('new_per_day')
        plt.xlabel('date')
        plt.plot(epidemiology_series[epidemiology_series['countrycode'] == country]['date'].values,
                 epidemiology_series[epidemiology_series['countrycode'] == country]['new_per_day'].values,
                 label='new_per_day')
        plt.plot(epidemiology_series[epidemiology_series['countrycode'] == country]['date'].values,
                 epidemiology_series[epidemiology_series['countrycode'] == country]['new_per_day_smooth'].values,
                 label='new_per_day_spline')
        plt.plot([epidemiology_series[epidemiology_series['countrycode'] == country]['date'].values[i]
                  for i in peak_characteristics[0]],
                 [epidemiology_series[epidemiology_series['countrycode'] == country]['new_per_day_smooth'].values[i]
                  for i in peak_characteristics[0]], "X", ms=20, color='red')
        plt.legend()
        plt.savefig(PLOT_PATH + 'epidemiological/' + country + '.png')
        plt.close()

government_response_panel = pd.DataFrame(columns=['countrycode', 'country', 'max_si','date_max_si','response_time',
                                                  'response_time_pop'] +
                                                 [flag + '_raised' for flag in flags] +
                                                 [flag + '_lowered' for flag in flags] +
                                                 [flag + '_raised_again' for flag in flags])

countries = government_response['countrycode'].unique()
for country in tqdm(countries,desc='Processing Gov Response Panel Data'):
    data = dict()
    data['countrycode'] = country
    data['country'] = government_response_series[government_response_series['countrycode'] == country]['country'].iloc[0]
    data['max_si'] = government_response_series[government_response_series['countrycode'] == country]['si'].max()
    data['date_max_si'] = government_response_series[government_response_series['si'] == data['max_si']]['date'].iloc[0]

    population = np.nan if len(wb_statistics[wb_statistics['countrycode'] == country]['value']) == 0 else \
        wb_statistics[wb_statistics['countrycode'] == country]['value'].iloc[0]
    t0 = np.nan if len(epidemiology_panel[epidemiology_panel['countrycode']==country]['t0']) == 0 \
        else epidemiology_panel[epidemiology_panel['countrycode']==country]['t0'].iloc[0]
    t0_relative = np.nan if len(epidemiology_panel[epidemiology_panel['countrycode']==country]['t0_relative']) == 0 \
        else epidemiology_panel[epidemiology_panel['countrycode']==country]['t0_relative'].iloc[0]

    data['response_time'] = np.nan if pd.isnull(t0) else (data['date_max_si'] - t0).days
    data['response_time_pop'] = np.nan if pd.isnull(t0_relative) else (data['date_max_si'] - t0_relative).days

    for flag in flags:
        days_above = pd.Series(
            government_response_series[
                government_response_series['countrycode'] == country][flag + '_days_above_threshold'])
        waves = [[cat[1], grp.shape[0]] for cat, grp in
                 days_above.groupby([days_above.ne(days_above.shift()).cumsum(), days_above])]

        data[flag + '_raised'] = np.nan
        data[flag + '_lowered'] = np.nan
        data[flag + '_raised_again'] = np.nan

        if len(waves) >= 2:
            data[flag + '_raised'] = government_response_series[
                government_response_series['countrycode'] == country]['date'].iloc[waves[0][1]]
        if len(waves) >= 3:
            data[flag + '_lowered'] = government_response_series[
                government_response_series['countrycode'] == country]['date'].iloc[
                waves[0][1] + waves[1][1]]
        if len(waves) >= 4:
            data[flag + '_lowered'] = government_response_series[
                government_response_series['countrycode'] == country]['date'].iloc[
                waves[0][1] + waves[1][1] + waves[2][1]]

    government_response_panel = government_response_panel.append(data,ignore_index=True)
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 3 - SAVING FIGURE 1a
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
    figure_1a.to_csv(CSV_PATH + 'figure_1a.csv')

# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 3 - SAVING FIGURE 1b
'''

figure_1b = epidemiology_panel[['countrycode','country','class']].dropna()
# It looks like pandas cannot correctly serialise the geometry column so this
# is being commented out. Possibly this merge could be done at plot time, or
# the data could be pickled instead of written to a flat CSV. Since it is just
# a left merge with the country code this should be feasible.
figure_1b = figure_1b.merge(map_data[['countrycode']],on='countrycode',how = 'left').dropna()
figure_1b = figure_1b.merge(map_data[['countrycode','geometry']],on='countrycode',how = 'left').dropna()

if SAVE_CSV:
    figure_1b.to_csv(CSV_PATH + 'figure_1b.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 4 - SAVING FIGURE 2
'''

data = epidemiology_series[['countrycode','country','date','days_since_t0','days_since_t0_pop']].merge(
    epidemiology_panel[['countrycode','class']], on='countrycode',how='left').merge(
    government_response_series[['countrycode','date','si']],on=['countrycode','date'],how='left').dropna()

figure_2 = pd.DataFrame(columns=['COUNTRYCODE','COUNTRY','CLASS','t','stringency_index'])
figure_2['COUNTRYCODE'] = data['countrycode']
figure_2['COUNTRY'] = data['country']
figure_2['CLASS'] = data['class']
figure_2['t'] = data['days_since_t0']
figure_2['t_pop'] = data['days_since_t0_pop']
figure_2['stringency_index'] = data['si']

if SAVE_CSV:
    figure_2.to_csv(CSV_PATH + 'figure_2.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 5 - SAVING FIGURE 3
'''

class_coarse = {
    0:np.nan,
    1:'EPI_FIRST_WAVE',
    2:'EPI_FIRST_WAVE',
    3:'EPI_SECOND_WAVE',
    4:'EPI_SECOND_WAVE'
}

data = epidemiology_panel[['countrycode', 'country', 'class', 'population', 'last_confirmed']]
data['last_confirmed_per_10k'] = 10000 * epidemiology_panel['last_confirmed'] / epidemiology_panel['population']
data['class'] = data['class'].apply(lambda x: class_coarse[x])
data = data.merge(government_response_panel[['countrycode', 'response_time','response_time_pop']],
                  how='left', on='countrycode').dropna()

figure_3 = pd.DataFrame(columns=['COUNTRYCODE', 'COUNTRY', 'GOV_MAX_SI_DAYS_FROM_T0',
                                 'CLASS_COARSE', 'POPULATION', 'EPI_CONFIRMED', 'EPI_CONFIRMED_PER_10K'])

figure_3['COUNTRYCODE'] = data['countrycode']
figure_3['COUNTRY'] = data['country']
figure_3['GOV_MAX_SI_DAYS_FROM_T0'] = data['response_time']
figure_3['GOV_MAX_SI_DAYS_FROM_T0_POP'] = data['response_time_pop']
figure_3['CLASS_COARSE'] = data['class']
figure_3['POPULATION'] = data['population']
figure_3['EPI_CONFIRMED'] = data['last_confirmed']
figure_3['EPI_CONFIRMED_PER_10K'] = data['last_confirmed_per_10k']

if SAVE_CSV:
    figure_3.to_csv(CSV_PATH + 'figure_3.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 6 - SAVING FIGURE 4
'''

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

# -------------------------------------------------------------------------------------------------------------------- #
'''
PART 6 - SAVING FIGURE 5
'''
figure_5 = epidemiology_series[[
    'country', 'countrycode', 'date', 'new_per_day', 'new_per_day_smooth',
    'dead_per_day', 'dead_per_day_smooth', 'new_tests', 'new_tests_smoothed', 'positive_rate']]

if SAVE_CSV:
    figure_5.to_csv(CSV_PATH + 'figure_5.csv')
# -------------------------------------------------------------------------------------------------------------------- #
'''
SAVING TIMESTAMP
'''

if SAVE_CSV:
    np.savetxt(CSV_PATH + 'last_updated.txt', [datetime.datetime.today().date().strftime('%Y-%m-%d')], fmt='%s')