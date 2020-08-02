import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import pandas as pd
import geopandas as gpd
import seaborn as sns
import os
import warnings
from tqdm import tqdm
import datetime
import re

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
assert not raw_epidemiology[['countrycode','date']].duplicated().any()

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

sql_command = """SELECT * FROM administrative_division WHERE adm_level=0"""
map_data = gpd.GeoDataFrame.from_postgis(sql_command, conn, geom_col='geometry')

##EPIDEMIOLOGY PRE-PROCESSING LOOP
countries = raw_epidemiology['countrycode'].unique()
epidemiology = pd.DataFrame(columns=['countrycode','country','date','confirmed','new_per_day','days_since_first'])
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
    days_since_first = np.zeros(len(data))
    placeholder_days_since_first = np.arange(1,len(data[data['confirmed']>0])+1)
    days_since_first[-len(placeholder_days_since_first)::] = placeholder_days_since_first
    data['days_since_first'] = days_since_first
    epidemiology = pd.concat((epidemiology,data)).reset_index(drop=True)
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

countries = np.union1d(epidemiology['countrycode'].unique(), government_response['countrycode'].unique())

exclude = []
for country in countries:
    if len(epidemiology[epidemiology['countrycode'] == country]) == 0 \
            or len(government_response[government_response['countrycode'] == country]) == 0:
        exclude.append(country)


epidemiology = epidemiology[~epidemiology['countrycode'].isin(exclude)]
epidemiology = epidemiology[epidemiology['countrycode'].isin(countries)]
government_response = government_response[~government_response['countrycode'].isin(exclude)]
government_response = government_response[government_response['countrycode'].isin(countries)]

LABELLED_COLUMNS = pd.read_csv('./peak_labels.csv')

CLASS_DICTIONARY = {
    'EPI_ENTERING_FIRST' : 1,
    'EPI_PAST_FIRST' : 2,
    'EPI_ENTERING_SECOND' : 3,
    'EPI_PAST_SECOND' : 4
}

classes = np.zeros(len(LABELLED_COLUMNS))
for k, v in CLASS_DICTIONARY.items():
    classes[np.where(LABELLED_COLUMNS[k])] += v
LABELLED_COLUMNS['CLASS'] = classes

epidemiology = epidemiology.merge(LABELLED_COLUMNS[['COUNTRYCODE','CLASS']], left_on = ['countrycode'],
                   right_on = ['COUNTRYCODE'], how = 'left')

flags_plotted = ['c1_school_closing', 'c2_workplace_closing', 'c6_stay_at_home_requirements']

def get_restriction_start_date(government_response,country,flag):
    conversion_dict={'c1_school_closing' : 3, 'c2_workplace_closing' : 2, 'c6_stay_at_home_requirements' : 2}
    temp = government_response[(government_response['countrycode'] == country) &
                               (government_response[flag] >= conversion_dict[flag])]
    if len(temp) == 0:
        return
    return temp.iloc[np.where(temp['date'].diff() != pd.Timedelta('1 days 00:00:00'))]['date'].values

def get_restriction_end_date(government_response,country,flag):
    conversion_dict={'c1_school_closing' : 3, 'c2_workplace_closing' : 2, 'c6_stay_at_home_requirements' : 2}
    temp = government_response[(government_response['countrycode'] == country) &
                               (government_response[flag] >= conversion_dict[flag])]
    if len(temp) == 0:
        return
    return temp.iloc[np.where(temp['date'].diff() != pd.Timedelta('1 days 00:00:00'))[0] - 1]['date'].values

for stage in list(CLASS_DICTIONARY.values())[0:2]:
    countries = epidemiology[epidemiology['CLASS'] == stage]['countrycode'].unique()

    c6_start_dates = [get_restriction_start_date(government_response, country, 'c6_stay_at_home_requirements')
                      for country in countries]

    c6_avg_start = list()
    for x in range(len(countries)):
        if len(c6_start_dates[x]) and \
                (len(epidemiology[(epidemiology['countrycode'] == countries[x]) &
                                  (epidemiology['date'] == c6_start_dates[x][0])]) > 0):
            c6_avg_start.append(epidemiology[(epidemiology['countrycode'] == countries[x]) &
                                             (epidemiology['date'] == c6_start_dates[x][0])]['days_since_first'].values[0])
    c6_mean_start = np.mean(c6_avg_start)
    c6_std_start = np.std(c6_avg_start)

    c6_end_dates = {country:get_restriction_end_date(government_response, country, 'c6_stay_at_home_requirements')
                      for country in countries}
    c6_avg_end = list()
    for x in range(len(countries)):
        if len(c6_end_dates[x]) > 1 and \
                (len(epidemiology[(epidemiology['countrycode'] == countries[x]) &
                                  (epidemiology['date'] == c6_end_dates[x][1])]) > 0):
            c6_avg_end.append(epidemiology[(epidemiology['countrycode'] == countries[x]) &
                                             (epidemiology['date'] == c6_end_dates[x][0])]['days_since_first'].values[0])
        if len(c6_end_dates[x]) == 1 and \
            (len(epidemiology[(epidemiology['countrycode'] == countries[x]) &
                                  (epidemiology['date'] == c6_end_dates[x][0])]) > 0):
            c6_avg_end.append(epidemiology[(epidemiology['countrycode'] == countries[x]) &
                                             (epidemiology['date'] == c6_end_dates[x][0])]['days_since_first'].values[0])

    c6_mean_end = np.mean(c6_avg_end)
    c6_std_end = np.std(c6_avg_end)

    countries = pd.Series({country:epidemiology[epidemiology['countrycode']==country]['confirmed'].iloc[-1]
                 for country in countries}).nlargest(n = 10).index.to_numpy()
    sns.set()
    plt.figure(figsize=(20, 7))
    for country in countries:
        ep_data = epidemiology[(epidemiology['countrycode'] == country) & (epidemiology['days_since_first'] > 0)]
        ep_data['new_per_day_7d_ma'] = ep_data['new_per_day'].rolling(7).mean() / ep_data['new_per_day'].max()
        sns.lineplot(x='days_since_first', y='new_per_day_7d_ma', data=ep_data, label=country)
        if stage == 1:
            plt.vlines([c6_mean_start - c6_std_start, c6_mean_start, c6_mean_start + c6_std_start],
                       0, 0.8, linestyles =['dashed','solid','dashed'])
            plt.fill_betweenx([0,0.8],[c6_mean_start - c6_std_start, c6_mean_start - c6_std_start],
                              [c6_mean_start + c6_std_start, c6_mean_start + c6_std_start],
                              facecolor = 'salmon', alpha=0.05)
        if stage == 2:
            plt.vlines([c6_mean_start - c6_std_start, c6_mean_start, c6_mean_start + c6_std_start],
                       0, 0.8, linestyles =['dashed','solid','dashed'])
            plt.fill_betweenx([0,0.8],[c6_mean_start - c6_std_start, c6_mean_start - c6_std_start],
                              [c6_mean_start + c6_std_start, c6_mean_start + c6_std_start],
                              facecolor = 'salmon', alpha=0.05)
            plt.vlines([c6_mean_end - c6_std_end, c6_mean_end, c6_mean_end + c6_std_end],
                       0, 0.8, linestyles =['dashed','solid','dashed'])
            plt.fill_betweenx([0,0.8],[c6_mean_end - c6_std_end, c6_mean_end - c6_std_end],
                              [c6_mean_end + c6_std_end, c6_mean_end + c6_std_end],
                              facecolor = 'salmon', alpha=0.05)

        if stage == 3:
            continue

        if stage == 4:
            continue