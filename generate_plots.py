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

'''Initialise script parameters'''

MIN_PERIOD = 14
PATH = './charts/table_figures/'

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
flags=['stringency_index','c1_school_closing','c2_workplace_closing','c6_stay_at_home_requirements']
flag_conversion = {'c1_school_closing': 2, 'c2_workplace_closing': 2, 'c6_stay_at_home_requirements': 2}

sql_command = """SELECT * FROM government_response"""
raw_government_response = pd.read_sql(sql_command, conn)
raw_government_response = raw_government_response.sort_values(by=['countrycode','date']).reset_index(drop=True)
raw_government_response = raw_government_response[['countrycode','country','date']+flags]
raw_government_response = raw_government_response.sort_values(by=['country','date'])\
    .drop_duplicates(subset=['countrycode','date'])#.dropna(subset=['stringency_index'])
raw_government_response = raw_government_response.sort_values(by=['countrycode','date'])
### Check no conflicting values for each country and date
assert not raw_government_response[['countrycode','date']].duplicated().any()

# GET COUNTRY STATS TABLE
required_stats = ['Population, total','Surface area (sq. km)']
sql_command = """SELECT * FROM world_bank"""
raw_country_stats = pd.read_sql(sql_command, conn)
raw_country_stats = raw_country_stats[raw_country_stats['indicator_name'].isin(required_stats)]

# GET ADMINISTRATIVE DIVISIONS
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

##COUNTRY STATS PRE-PROCESSING LOOP
countries = raw_country_stats['countrycode'].unique()
country_stats = pd.DataFrame(columns = ['countrycode','country']+required_stats)
for country in countries:
    data = raw_country_stats[raw_country_stats['countrycode']==country]
    data = data[['countrycode','country','indicator_name','value']]
    stats = {k:0 for k in required_stats}
    for stat in required_stats:
        stats[stat] = [data[data['indicator_name']==stat]['value'].values[0]]
    stats['countrycode'] = [data['countrycode'].iloc[0]]
    stats['country'] = [data['country'].iloc[0]]
    stats = pd.DataFrame.from_dict(stats)
    country_stats = pd.concat((country_stats,stats)).reset_index(drop=True)


countries = np.union1d(epidemiology['countrycode'].unique(),
                       np.union1d(government_response['countrycode'].unique(),country_stats['countrycode'].unique()))

exclude = []
for country in countries:
    if len(epidemiology[epidemiology['countrycode'] == country]) == 0 \
            or len(government_response[government_response['countrycode'] == country]) == 0\
            or len(country_stats[country_stats['countrycode'] == country]) == 0:
        exclude.append(country)


epidemiology = epidemiology[~epidemiology['countrycode'].isin(exclude)]
epidemiology = epidemiology[epidemiology['countrycode'].isin(countries)]
government_response = government_response[~government_response['countrycode'].isin(exclude)]
government_response = government_response[government_response['countrycode'].isin(countries)]
country_stats = country_stats[~country_stats['countrycode'].isin(exclude)]
country_stats = country_stats[country_stats['countrycode'].isin(countries)]

## Processing epidemiology

EPI = {
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'date':np.empty(0),
    'confirmed':np.empty(0),
    'class':np.empty(0),
    'new_per_day':np.empty(0),
    'new_per_day_per_10k':np.empty(0),
    'days_since_first':np.empty(0),
    'days_since_30_per_day':np.empty(0)
}

for country in countries:
    data = epidemiology[epidemiology['countrycode']==country]
    if len(country_stats[country_stats['countrycode']==country]) == 0 or len(data[data['new_per_day']>30]['date']) == 0:
        continue
    EPI['countrycode'] = np.concatenate((EPI['countrycode'], data['countrycode'].values))
    EPI['country'] = np.concatenate((EPI['country'], data['country'].values))
    EPI['date'] = np.concatenate((EPI['date'], data['date'].values))
    EPI['confirmed'] = np.concatenate((EPI['confirmed'], data['confirmed'].values))
    EPI['class'] = np.concatenate((EPI['class'], data['CLASS'].values))
    EPI['new_per_day'] = np.concatenate((EPI['new_per_day'], data['new_per_day'].values))
    EPI['days_since_first'] = np.concatenate((EPI['days_since_first'], data['days_since_first'].values))

    EPI['new_per_day_per_10k'] = np.concatenate((
        EPI['new_per_day_per_10k'],
        (data['new_per_day'].values /
         country_stats[country_stats['countrycode']==country]['Population, total'].values) * 10000))

    #np.max(data['new_per_day'].values)

    date_of_30 = data[data['new_per_day']>30]['date'].iloc[0]
    EPI['days_since_30_per_day'] = np.concatenate((EPI['days_since_30_per_day'],
                                                   np.concatenate((np.arange(-len(data[data['date']<date_of_30]),0),
                                                                   np.arange(len(data[data['date']>=date_of_30]))))))

EPI = pd.DataFrame.from_dict(EPI)

## Processing Gov Response

GOV = {
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'last_confirmed':np.empty(0),
}

for flag in flag_conversion.keys():
    GOV[flag + '_date_raised'] = np.empty(0)
    GOV[flag + '_date_lowered'] = np.empty(0)
    GOV[flag + '_date_raised_again'] = np.empty(0)
    GOV[flag + '_more?'] = np.empty(0)
    GOV[flag + '_days_since_30_raised'] = np.empty(0)
    GOV[flag + '_days_since_30_lowered'] = np.empty(0)
    GOV[flag + '_days_since_30_raised_again'] = np.empty(0)
    GOV[flag + '_date_raised_cases_day'] = np.empty(0)
    GOV[flag + '_date_lowered_cases_day'] = np.empty(0)
    GOV[flag + '_date_raised_again_cases_day'] = np.empty(0)

countries = EPI['countrycode'].unique()
for country in countries:
    data = government_response[government_response['countrycode']==country]
    if (len(data) == 0) or (len(epidemiology[epidemiology['countrycode']==country]) == 0):
        continue
    GOV['countrycode'] = np.concatenate((GOV['countrycode'], [data['countrycode'].values[0]]))
    GOV['country'] = np.concatenate((GOV['country'], [data['country'].values[0]]))
    GOV['last_confirmed'] = np.concatenate((GOV['last_confirmed'],
                                            [epidemiology[epidemiology['countrycode']==country]['confirmed'].iloc[-1]]))
    for flag in flag_conversion.keys():
        days_above = (data[flag] >= flag_conversion[flag]).astype(int)
        waves = [[cat[1], grp.shape[0]] for cat, grp in
                 days_above.groupby([days_above.ne(days_above.shift()).cumsum(), days_above])]

        for i, wave in enumerate(waves):
            if wave[0] == 0:
                continue
            if wave[1] < MIN_PERIOD:
                waves[i-1][1] += waves[i][1]
                del waves[i]

        if len(waves) <= 1:
            GOV[flag + '_date_raised'] = np.concatenate((GOV[flag + '_date_raised'], [np.nan]))
            GOV[flag + '_date_lowered'] = np.concatenate((GOV[flag + '_date_lowered'], [np.nan]))
            GOV[flag + '_date_raised_again'] = np.concatenate((GOV[flag + '_date_raised_again'], [np.nan]))
            GOV[flag + '_more?'] = np.concatenate((GOV[flag + '_more?'], [np.nan]))
            GOV[flag + '_days_since_30_raised'] = np.concatenate((GOV[flag + '_days_since_30_raised'], [np.nan]))
            GOV[flag + '_days_since_30_lowered'] = np.concatenate((GOV[flag + '_days_since_30_lowered'], [np.nan]))
            GOV[flag + '_days_since_30_raised_again'] = np.concatenate((GOV[flag + '_days_since_30_raised_again'], [np.nan]))
            GOV[flag + '_date_raised_cases_day'] = np.concatenate((GOV[flag + '_date_raised_cases_day'], [np.nan]))
            GOV[flag + '_date_lowered_cases_day'] = np.concatenate((GOV[flag + '_date_lowered_cases_day'], [np.nan]))
            GOV[flag + '_date_raised_again_cases_day'] = np.concatenate((GOV[flag + '_date_raised_again_cases_day'], [np.nan]))
            continue

        date_raised = data['date'].iloc[waves[0][1] - 1] \
            if data['date'].iloc[waves[0][1] - 1] != None else np.nan
        date_lowered = data['date'].iloc[waves[1][1]+waves[0][1] - 1] \
            if data['date'].iloc[waves[1][1]+waves[0][1] - 1] != None else np.nan
        date_raised_again = data['date'].iloc[sum([waves[x][1] for x in range(3)]) - 1] \
            if len(waves) >= 4 else np.nan
        more = True if len(waves) >= 5 else False

        days_since_30_raised = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised)]['days_since_30_per_day'].values
        days_since_30_raised = [np.nan] if len(days_since_30_raised) == 0 else days_since_30_raised

        days_since_30_lowered = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_lowered)]['days_since_30_per_day'].values
        days_since_30_lowered = [np.nan] if len(days_since_30_lowered) == 0 else days_since_30_lowered

        days_since_30_raised_again = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised_again)]['days_since_30_per_day'].values \
                if len(waves) >= 4 else [np.nan]
        days_since_30_raised_again = [np.nan] if len(days_since_30_raised_again) == 0 else days_since_30_raised_again

        date_raised_cases_day = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised)]['new_per_day'].values
        date_raised_cases_day = [np.nan] if len(date_raised_cases_day) == 0 else date_raised_cases_day

        date_lowered_cases_day = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_lowered)]['new_per_day'].values
        date_lowered_cases_day = [np.nan] if len(date_lowered_cases_day) == 0 else date_lowered_cases_day

        date_raised_again_cases_day = \
            EPI[(EPI['countrycode'] == country) & (EPI['date'] == date_raised_again)]['new_per_day'].values \
                if len(waves) >= 4 else [np.nan]
        date_raised_again_cases_day = np.nan if len(date_raised_again_cases_day) == 0 else date_raised_again_cases_day

        GOV[flag + '_date_raised'] = np.concatenate((GOV[flag + '_date_raised'], [date_raised]))
        GOV[flag + '_date_lowered'] = np.concatenate((GOV[flag + '_date_lowered'], [date_lowered]))
        GOV[flag + '_date_raised_again'] = np.concatenate((GOV[flag + '_date_raised_again'], [date_raised_again]))
        GOV[flag + '_more?'] = np.concatenate((GOV[flag + '_more?'], [more]))
        GOV[flag + '_days_since_30_raised'] = np.concatenate((GOV[flag + '_days_since_30_raised'], days_since_30_raised))
        GOV[flag + '_days_since_30_lowered'] = np.concatenate((GOV[flag + '_days_since_30_lowered'], days_since_30_lowered))
        GOV[flag + '_days_since_30_raised_again'] = np.concatenate(
            (GOV[flag + '_days_since_30_raised_again'], days_since_30_raised_again))
        GOV[flag + '_date_raised_cases_day'] = np.concatenate((GOV[flag + '_date_raised_cases_day'], date_raised_cases_day))
        GOV[flag + '_date_lowered_cases_day'] = np.concatenate((GOV[flag + '_date_lowered_cases_day'], date_lowered_cases_day))
        GOV[flag + '_date_raised_again_cases_day'] = np.concatenate(
            (GOV[flag + '_date_raised_again_cases_day'], date_raised_again_cases_day))

GOV = pd.DataFrame.from_dict(GOV)
GOV = GOV.merge(EPI.drop_duplicates(subset=['countrycode','class'])
                [['countrycode','class']], on = ['countrycode'], how = 'left')
GOV['class'] = GOV['class'].astype(int)

f, ax = plt.subplots(figsize=(20, 7))
plt.ylim(0,0.002)
sns.distplot(GOV['c6_stay_at_home_requirements_date_raised_cases_day'], bins = 500, hist=False)
plt.savefig(PATH + 'flag_c6_raised_new_cases_per_day.png')

chosen_flag = 'c6_stay_at_home_requirements'

for cls in GOV['class'].unique():
    if cls == 0 or cls == 4:
        continue

    plt.figure(figsize=(20,7))
    countries = EPI[EPI['class'] == cls]['countrycode'].unique()

    avg_raised = np.mean(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_raised'])
    std_raised = np.std(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_raised'])
    n1 = len(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_raised'].dropna())

    avg_lowered = np.mean(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_lowered'])
    std_lowered = np.std(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_lowered'])
    n2 = len(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_lowered'].dropna())

    avg_raised_again = np.mean(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_raised_again'])
    std_raised_again = np.std(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_raised_again'])
    n3 = len(GOV[GOV['countrycode'].isin(countries)][chosen_flag+'_days_since_30_raised_again'].dropna())

    countries = pd.Series({country:EPI[EPI['countrycode']==country]['confirmed'].iloc[-1]
                           for country in countries}).nlargest(n = 10).index.to_numpy()
    aggregate = []
    for country in countries:
        data = EPI[(EPI['countrycode'] == country) & (EPI['days_since_30_per_day'] >= 0)]
        data['new_per_day_7d_ma'] = data['new_per_day_per_10k'].rolling(7).mean() / data['new_per_day_per_10k'].max()
        aggregate.append(data['new_per_day_7d_ma'].values)
        sns.lineplot(x='days_since_30_per_day',y='new_per_day_7d_ma',data=data, label=country)
        continue

    aggregate = np.array([y[0:np.min([len(x) for x in aggregate])] for y in aggregate])
    aggregate = np.nanmean(aggregate, axis=0)
    sns.lineplot(x = np.arange(len(aggregate)),y = aggregate, color = 'black', linewidth = 5, label = 'aggregate')

    if cls == 1:
        plt.vlines([avg_raised - std_raised, avg_raised, avg_raised + std_raised], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised - std_raised, avg_raised - std_raised],
                      [avg_raised + std_raised, avg_raised + std_raised],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised, 0.5,'Flag Raised', rotation = 90)
        plt.text(avg_raised, 1,'n = ' + str(n1))

    if cls == 2:
        plt.vlines([avg_raised - std_raised, avg_raised, avg_raised + std_raised], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised - std_raised, avg_raised - std_raised],
                      [avg_raised + std_raised, avg_raised + std_raised],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised, 0.5, 'Flag Raised', rotation = 90)
        plt.text(avg_raised, 1, 'n = ' + str(n1))

        plt.vlines([avg_lowered - std_lowered, avg_lowered, avg_lowered + std_lowered], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_lowered - std_lowered, avg_lowered - std_lowered],
                      [avg_lowered + std_lowered, avg_lowered + std_lowered],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_lowered, 0.5, 'Flag Lowered', rotation = 90)
        plt.text(avg_lowered, 1, 'n = ' + str(n2))

    if cls == 3:
        plt.vlines([avg_raised - std_raised, avg_raised, avg_raised + std_raised], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised - std_raised, avg_raised - std_raised],
                      [avg_raised + std_raised, avg_raised + std_raised],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised, 0.5, 'Flag Raised', rotation = 90)
        plt.text(avg_raised, 1, 'n = ' + str(n1))

        plt.vlines([avg_lowered - std_lowered, avg_lowered, avg_lowered + std_lowered], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_lowered - std_lowered, avg_lowered - std_lowered],
                      [avg_lowered + std_lowered, avg_lowered + std_lowered],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_lowered, 0.5, 'Flag Lowered', rotation = 90)
        plt.text(avg_lowered, 1, 'n = ' + str(n2))

        plt.vlines([avg_raised_again - std_raised_again, avg_raised_again, avg_raised_again + std_raised_again], 0, 1,
                   color = 'black', linestyles=['dashed', 'solid', 'dashed'])
        plt.fill_betweenx([0, 1], [avg_raised_again - std_raised_again, avg_raised_again - std_raised_again],
                      [avg_raised_again + std_raised_again, avg_raised_again + std_raised_again],
                      facecolor='salmon', alpha=0.3)
        plt.text(avg_raised_again, 0.5, 'Flag Raised Again', rotation = 90)
        plt.text(avg_raised_again, 1, 'n = ' + str(n3))

    plt.legend()
    plt.savefig(PATH + 'stage_' + str(cls) + '_timeline.png')