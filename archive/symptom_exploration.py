import numpy as np
import pandas as pd
import requests
import json
import psycopg2
from tqdm import tqdm
from csaps import csaps
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.signal import find_peaks
import datetime
from sklearn.linear_model import LinearRegression

# SET SCRIPT PARAMETERS

LOAD_CSV = True
SMOOTH = 0.001
T0_THRESHOLD = 1000
PROMINENCE_THRESHOLD = 5
DISTANCE = 21

# LOAD EPIDEMIOLOGY DATA

conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19')
cur = conn.cursor()

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
mobilities = ['residential','workplace','transit_stations']

sql_command = """SELECT * FROM mobility WHERE source = %(source)s AND adm_area_1 is NULL"""
raw_mobility = pd.read_sql(sql_command, conn, params={'source': source})
raw_mobility = raw_mobility[['countrycode', 'country', 'date'] + mobilities]
raw_mobility = raw_mobility.sort_values(by=['countrycode', 'date']).reset_index(drop=True)
# Check no conflicting values for each country and date
assert not raw_mobility[['countrycode', 'date']].duplicated().any()

# GET RAW GOVERNMENT RESPONSE TABLE
flags = ['c6_stay_at_home_requirements','c2_workplace_closing','c5_close_public_transport']
flag_thresholds = {'c6_stay_at_home_requirements': 2, 'c2_workplace_closing':2, 'c5_close_public_transport':1}
flag_to_mobility = {'c6_stay_at_home_requirements': 'residential'}

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
map_data = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')[['countrycode','geometry']]


indicator_code = 'SP.POP.TOTL'
sql_command = """SELECT countrycode, value, year FROM world_bank WHERE 
adm_area_1 IS NULL AND indicator_code = %(indicator_code)s"""
wb_statistics = pd.read_sql(sql_command, conn, params={'indicator_code': indicator_code})
assert len(wb_statistics) == len(wb_statistics['countrycode'].unique())
wb_statistics = wb_statistics.sort_values(by=['countrycode', 'year'], ascending=[True, False]).reset_index(drop=True)

# EPIDEMIOLOGY PROCESSING
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode', 'country', 'date', 'confirmed', 'new_per_day'])
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

epidemiology_series = {
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0),
    'confirmed': np.empty(0),
    'new_per_day': np.empty(0),
    'new_per_day_smooth': np.empty(0),
    'dead': np.empty(0),
    'days_since_t0': np.empty(0),
    'new_cases_per_10k': np.empty(0)
}

mobility_series = {
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0)
}

for mobility_type in mobilities:
    mobility_series[mobility_type] = np.empty(0)
    mobility_series[mobility_type + '_smooth'] = np.empty(0)

government_response_series = {
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'date': np.empty(0),
    'si': np.empty(0)
}

for flag in flags:
    government_response_series[flag] = np.empty(0)
    government_response_series[flag + '_days_above_threshold'] = np.empty(0)

countries = np.sort(epidemiology['countrycode'].unique())
for country in tqdm(countries, desc='Processing Epidemiological Time Series Data'):
    data = epidemiology[epidemiology['countrycode'] == country]

    x = np.arange(len(data['date']))
    y = data['new_per_day'].values
    ys = csaps(x, y, x, smooth=SMOOTH)

    t0 = np.nan if len(data[data['confirmed']>T0_THRESHOLD]['date']) == 0 else \
        data[data['confirmed']>T0_THRESHOLD]['date'].iloc[0]
    population = np.nan if len(wb_statistics[wb_statistics['countrycode']==country]['value'])==0 else \
        wb_statistics[wb_statistics['countrycode']==country]['value'].iloc[0]
    days_since_t0 = np.repeat(np.nan,len(data)) if pd.isnull(t0) else \
        np.array([(date - t0).days for date in data['date'].values])
    new_cases_per_10k = 10000 * (ys / population)

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
    epidemiology_series['days_since_t0'] = np.concatenate(
        (epidemiology_series['days_since_t0'], days_since_t0))
    epidemiology_series['new_cases_per_10k'] = np.concatenate(
        (epidemiology_series['new_cases_per_10k'], new_cases_per_10k))



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
# LOAD SYMPTOM DATA

if not LOAD_CSV:
    covid_symptom_series = pd.DataFrame(columns=['country', 'gid_0', 'iso_code', 'sample_size',
                                           'smoothed_cli', 'smoothed_cli_se', 'survey_date'])
    flu_symptom_series = pd.DataFrame(columns=['country', 'gid_0', 'iso_code', 'sample_size',
                                           'smoothed_ili', 'smoothed_ili_se', 'survey_date'])

    countries_available_url = 'https://covidmap.umd.edu/api/country'
    response = requests.get(countries_available_url).text
    jsonData = json.loads(response)
    countries_available = [x['country'] for x in jsonData['data']]

    for country in tqdm(countries_available):
        dates_available_url = 'https://covidmap.umd.edu/api/datesavail?country={}'.format(country)
        response = requests.get(dates_available_url).text
        jsonData = json.loads(response)
        start_date, end_date = jsonData['data'][0]['survey_date'], jsonData['data'][-1]['survey_date']

        covid_url = 'https://covidmap.umd.edu/api/resources?indicator=covid&type=smoothed&country={}&daterange={}-{}'\
            .format(country,start_date,end_date)
        response = requests.get(covid_url).text
        jsonData = json.loads(response)['data']

        for data_date in jsonData:
            covid_symptom_series = covid_symptom_series.append(data_date,ignore_index=True)

        flu_url = 'https://covidmap.umd.edu/api/resources?indicator=flu&type=smoothed&country={}&daterange={}-{}'\
            .format(country,start_date,end_date)
        response = requests.get(flu_url).text
        jsonData = json.loads(response)['data']

        for data_date in jsonData:
            flu_symptom_series = flu_symptom_series.append(data_date,ignore_index=True)

    symptom_series = pd.merge(covid_symptom_series, flu_symptom_series, how='inner',
             on=['country', 'gid_0', 'iso_code', 'survey_date'], suffixes=['_covid','_flu'])
    symptom_series['survey_date'] = pd.to_datetime(symptom_series['survey_date'])
    symptom_series = symptom_series[[
        'iso_code', 'sample_size_covid', 'smoothed_cli', 'smoothed_cli_se', 'survey_date', 'sample_size_flu',
        'smoothed_ili', 'smoothed_ili_se']].rename(columns={'iso_code': 'countrycode', 'survey_date': 'date'})
    symptom_series['date'] = symptom_series['date'].apply(lambda x: x.date())
    symptom_series.to_csv('./archive/symptom_data.csv')
else:
    symptom_series = pd.read_csv('./archive/symptom_data.csv', index_col=0, parse_dates=['date'])
    symptom_series = symptom_series[[
        'countrycode', 'sample_size_covid', 'smoothed_cli', 'smoothed_cli_se', 'date', 'sample_size_flu',
        'smoothed_ili', 'smoothed_ili_se']]
    symptom_series['date'] = symptom_series['date'].apply(lambda x: x.date())
# -------------------------------------------------------------------------------------------------------------------- #
# GET COUNTRY CLASS

class_dictionary = {
    'EPI_ENTERING_FIRST': 1,
    'EPI_PAST_FIRST': 2,
    'EPI_ENTERING_SECOND': 3,
    'EPI_PAST_SECOND': 4}
labelled_columns = pd.read_csv('./peak_labels.csv')

labelled_columns['class'] = pd.Series(0 if np.sum(labelled_columns[labelled_columns['COUNTRYCODE']==country][[
        'EPI_ENTERING_FIRST', 'EPI_PAST_FIRST', 'EPI_ENTERING_SECOND', 'EPI_PAST_SECOND']].values) == 0 else \
        class_dictionary[labelled_columns[labelled_columns['COUNTRYCODE']==country][[
        'EPI_ENTERING_FIRST', 'EPI_PAST_FIRST', 'EPI_ENTERING_SECOND', 'EPI_PAST_SECOND']].idxmax(axis=1).values[0]]
                                      for country in labelled_columns['COUNTRYCODE'].unique())
# -------------------------------------------------------------------------------------------------------------------- #
# GET TESTING DATA

raw_testing_series = pd.read_csv('./archive/owid-covid-data.csv', parse_dates=['date'])[[
    'iso_code', 'date', 'positive_rate']].rename(columns={'iso_code':'countrycode'})
testing_series = pd.DataFrame(columns=['countrycode','date','positive_rate'])

countries = [country for country in raw_testing_series['countrycode'].unique()
             if not(pd.isnull(country)) and (len(country) == 3)]
for country in countries:
    data = raw_testing_series[raw_testing_series['countrycode'] == country].reset_index()
    if len(data['positive_rate'].dropna()) == 0:
        continue
    data = data.iloc[data['positive_rate'][data['positive_rate'].notnull()].index[0]:
                     data['positive_rate'][data['positive_rate'].notnull()].index[-1]]
    testing_series = pd.concat((testing_series, data), ignore_index=True)

testing_series = testing_series[['countrycode', 'date', 'positive_rate']]
testing_series['date'] = pd.Series(x.date() for x in testing_series['date'])
# -------------------------------------------------------------------------------------------------------------------- #
# ANALYSIS - SYMPTOM

'''
epidemiology_series new_per_day_smooth
symptom_series smooth_cli, smooth_cli_se, smooth_ili, smooth_ili_se
'''

data = pd.merge(symptom_series, epidemiology_series, on=['countrycode','date'], how='left')\
    .merge(labelled_columns[['COUNTRYCODE','class']], left_on='countrycode', right_on='COUNTRYCODE', how='inner')\
    .merge(testing_series, on=['countrycode','date'], how='left')


countries = data[(data['class'] == 3) |(data['class'] == 4)]['countrycode'].unique()
"""
for country in countries:
    date_series = data[data['countrycode'] == country]['date'].values
    new_per_day = data[data['countrycode'] == country]['new_per_day_smooth'].values
    new_per_day /= np.max(new_per_day)

    covid_like = data[data['countrycode'] == country]['smoothed_cli'].values
    flu_like = data[data['countrycode'] == country]['smoothed_ili'].values
    covid_like /= np.max(np.concatenate((covid_like, flu_like)))
    flu_like /= np.max(np.concatenate((covid_like, flu_like)))

    positive_rate = data[data['countrycode'] == country]['positive_rate'].values
    positive_rate /= np.max(positive_rate)

    plt.figure()
    plt.plot(date_series, new_per_day, label = 'new_per_day')
    plt.plot(date_series, covid_like, label = 'covid_like_symptoms')
    plt.plot(date_series, flu_like, label = 'flu_like_symptoms')
    plt.plot(date_series, positive_rate, label = 'positive_rate')
    plt.title(country)
    plt.legend()
    plt.savefig('./archive/symptom_plots/' + country + '.png')
    plt.close()
"""
# -------------------------------------------------------------------------------------------------------------------- #
# GET PANEL DATA

epidemiology_panel = pd.DataFrame(columns=['countrycode', 'country', 'class', 'population', 'T0', 'peak_1', 'peak_2',
                                           'date_peak_1', 'date_peak_2', 'first_wave_start', 'first_wave_end',
                                           'second_wave_start', 'second_wave_end','last_confirmed'])

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
    data['T0'] = np.nan if len(epidemiology_series[(epidemiology_series['countrycode']==country) &
                                     (epidemiology_series['confirmed']>=T0_THRESHOLD)]['date']) == 0 else \
        epidemiology_series[(epidemiology_series['countrycode']==country) &
                                     (epidemiology_series['confirmed']>=T0_THRESHOLD)]['date'].iloc[0]

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
    epidemiology_panel = epidemiology_panel.append(data,ignore_index=True)

government_response_panel = pd.DataFrame(columns=['countrycode', 'country', 'max_si','date_max_si','response_time'] +
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
    t0 = np.nan if len(epidemiology_panel[epidemiology_panel['countrycode']==country]['T0']) == 0 \
        else epidemiology_panel[epidemiology_panel['countrycode']==country]['T0'].iloc[0]
    data['response_time'] = np.nan if pd.isnull(t0) else (data['date_max_si'] - t0).days

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
# ANALYSIS - DIFF IN DIFF - LINEAR REGRESSION COUNTER-FACTUAL
'''
14 days before - 14 days after
'''
country = 'USA'
flag_date = government_response_panel[
    government_response_panel['countrycode'] == country][flags[0] + '_raised'].values[0]
start_date = flag_date - datetime.timedelta(days=14)
end_date = flag_date + datetime.timedelta(days=14)

ep = epidemiology_series[(epidemiology_series['countrycode'] == country) &
                         (epidemiology_series['date'] >= start_date) &
                         (epidemiology_series['date'] <= end_date)]['new_per_day_smooth']

gov = government_response_series[(government_response_series['countrycode'] == country) &
                                 (government_response_series['date'] >= start_date) &
                                 (government_response_series['date'] <= end_date)]['si']

mob = mobility_series[(mobility_series['countrycode'] == country) &
                      (mobility_series['date'] >= start_date) &
                      (mobility_series['date'] <= end_date)][flag_to_mobility[flags[0]] + '_smooth']

model = LinearRegression()
model.fit(np.arange(len(mob) // 2).reshape((-1, 1)), mob.values[0:(len(mob) // 2)])
mob_extrapolated = model.predict(np.arange(len(mob)).reshape((-1, 1)))

plt.figure(figsize=(20,7))
#plt.plot(pd.date_range(start_date,end_date), ep / np.max(ep), label='new_per_day')
plt.plot(pd.date_range(start_date,end_date), gov, label='si')
plt.plot(pd.date_range(start_date,end_date), mob, label='residential')
plt.plot(pd.date_range(start_date,end_date), mob_extrapolated, linestyle='dashed', color='black')
plt.vlines(flag_date, 0, np.max(mob),linestyles='dashed',colors='red')
plt.legend()

# -------------------------------------------------------------------------------------------------------------------- #
# AGE DISTRIBUTION IN SECOND WAVE

bel_patients = pd.read_csv('./archive/COVID19BE_CASES_AGESEX.csv', parse_dates=['DATE']).dropna()
age_groups = np.sort(bel_patients['AGEGROUP'].unique()[~pd.isna(bel_patients['AGEGROUP'].unique())])
sex_groups = np.sort(bel_patients['SEX'].unique()[~pd.isna(bel_patients['SEX'].unique())])
age_dist = bel_patients[['DATE','AGEGROUP','CASES']].groupby(['DATE','AGEGROUP'], as_index=False).sum()

emptyframe = pd.DataFrame()
emptyframe['DATE'] = bel_patients['DATE'].unique().repeat(len(age_groups))
emptyframe['AGEGROUP'] = np.tile(age_groups,len(bel_patients['DATE'].unique()))
emptyframe['CASES'] = np.zeros(len(np.repeat(age_groups,len(bel_patients['DATE'].unique()))))

age_dist = emptyframe.merge(age_dist,
                 on=['DATE','AGEGROUP'], how='left', suffixes=['_x','']).fillna(0)[['DATE', 'AGEGROUP', 'CASES']]

plot_key = {k:v for k,v in zip(age_groups, range(len(age_groups)))}
lower_band = list()
median_age = list()
upper_band = list()
for date in age_dist['DATE'].unique():
    data = age_dist[age_dist['DATE'] == date]
    median = [data['AGEGROUP'].values[i] for i in
              range(len(data['AGEGROUP'].values)) for j in range(int(data['CASES'].values[i]))]
    median_age.append(median[len(median)//2])
    lower_band.append(median[len(median)//4])
    upper_band.append(median[3*len(median)//4])

plt.plot(age_dist['DATE'].unique(), [plot_key[i] for i in median_age], color='steelblue', label = 'median')
plt.fill_between(age_dist['DATE'].unique(),
                 [plot_key[i] for i in lower_band],
                 [plot_key[i] for i in upper_band], alpha=0.2)

plt.yticks(range(len(age_groups)), age_groups)
plt.legend()
# -------------------------------------------------------------------------------------------------------------------- #
