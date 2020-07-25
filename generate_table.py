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

import statsmodels.api as sm
from scipy.ndimage.interpolation import shift
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
raw_mobility = raw_mobility[['countrycode','country','date']+mobilities]
raw_mobility = raw_mobility.sort_values(by=['countrycode','date']).reset_index(drop=True)
### Check no conflicting values for each country and date
assert not raw_mobility[['countrycode','country','date']].duplicated().any()

# GET GOVERNMENT RESPONSE TABLE
flags=['stringency_index','c1_school_closing','c2_workplace_closing','c3_cancel_public_events',
       'c4_restrictions_on_gatherings','c5_close_public_transport','c6_stay_at_home_requirements',
       'c7_restrictions_on_internal_movement']

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

##EPIDEMIOLOGY PRE-PROCESSING LOOP
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

if SAVE_PLOTS:
    os.makedirs(PATH+'epidemiological/',exist_ok=True)

##MOBILITY TABLE
mobility_columns={
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'date':np.empty(0)
    }

if SAVE_PLOTS:
    os.makedirs(PATH+'mobility/',exist_ok=True)

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

percentiles = [25, 50, 75]
government_response_columns={
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'max_si':np.empty(0),
    'max_si_start_date':np.empty(0),
    'max_si_end_date':np.empty(0),
    'max_si_duration':np.empty(0),
    'max_si_currently':np.empty(0),
    'multiple_peaks':np.empty(0),
    'peak_heights':np.empty(0),
    'peak_start_date':np.empty(0),
    'peak_end_date':np.empty(0),
    'peak_widths':np.empty(0),
    'peak_prominences':np.empty(0),
    'high_restrictions_start_date':np.empty(0),
    'high_restrictions_end_date':np.empty(0),
    'high_restrictions_duration':np.empty(0),
    'high_restrictions_current':np.empty(0),
    'stringency_index_opt_lag':np.empty(0)
}

for percentile in percentiles:
    government_response_columns[str(percentile) + '_duration'] = np.empty(0)

for flag in flags:
    government_response_columns[flag + '_opt_lag'] = np.empty(0)

if SAVE_PLOTS:
    os.makedirs(PATH+'government_response/',exist_ok=True)

'''
EPIDEMIOLOGY PROCESSING
'''

countries = np.sort(epidemiology['countrycode'].unique())

for country in tqdm(countries, desc = 'Processing Epidemiological Data'):
    data = epidemiology[epidemiology['countrycode'] == country]
    data['new_per_day_ma'] = data['new_per_day'].rolling(ROLLING_WINDOW).mean().fillna(value=0)

    ###Fitting the spline
    x = np.arange(len(data['date']))
    y = data['new_per_day_ma'].values
    ys = csaps(x, y, x, smooth = SMOOTH)

    ###Finding peaks
    peak_locations = find_peaks(ys,distance=DISTANCE)[0]
    peak_mask = np.zeros(len(data['date']))
    for i in range(len(peak_locations)):
        peak_mask[peak_locations[i]] = i + 1
    peak_heights = np.array([ys[i] if peak_mask[i] != 0 else 0 for i in range(len(data['date']))])

    widths, width_heights, width_left, width_right = peak_widths(ys, peak_locations, rel_height=1)
    peak_width_values_mask = np.zeros(len(data['date']))
    peak_width_dates_mask = np.zeros(len(data['date']))
    peak_width_heights_mask = np.zeros(len(data['date']))

    prominences = peak_prominences(ys,peak_locations)[0]
    peak_prominence_mask = np.zeros(len(data['date']))

    for i in range(len(peak_locations)):
        peak_width_values_mask[peak_locations[i]] = widths[i]
        peak_width_heights_mask[peak_locations[i]] = width_heights[i]
        peak_width_dates_mask[int(np.rint(width_left[i])):int(np.rint(width_right[i]))+1] = i+1
        peak_prominence_mask[peak_locations[i]] = prominences[i]

    ###Upserting data
    epidemiology_columns['countrycode']=np.concatenate((
        epidemiology_columns['countrycode'],data['countrycode'].values))
    epidemiology_columns['country'] = np.concatenate(
        (epidemiology_columns['country'], data['country'].values))
    epidemiology_columns['date'] = np.concatenate(
        (epidemiology_columns['date'], data['date'].values))
    epidemiology_columns['confirmed'] = np.concatenate(
        (epidemiology_columns['confirmed'], data['confirmed'].values))
    epidemiology_columns['new_per_day'] = np.concatenate(
        (epidemiology_columns['new_per_day'], data['new_per_day'].values))
    epidemiology_columns['new_per_day_ma'] = np.concatenate(
        (epidemiology_columns['new_per_day_ma'], data['new_per_day_ma'].values))
    epidemiology_columns['new_per_day_smooth'] = np.concatenate(
        (epidemiology_columns['new_per_day_smooth'], ys))
    epidemiology_columns['peak_dates'] = np.concatenate(
        (epidemiology_columns['peak_dates'], peak_mask))
    epidemiology_columns['peak_heights'] = np.concatenate(
        (epidemiology_columns['peak_heights'], peak_heights))
    epidemiology_columns['peak_widths'] = np.concatenate(
        (epidemiology_columns['peak_widths'], peak_width_values_mask))
    epidemiology_columns['peak_prominence'] = np.concatenate(
        (epidemiology_columns['peak_prominence'], peak_prominence_mask))
    epidemiology_columns['peak_width_boundaries'] = np.concatenate(
        (epidemiology_columns['peak_width_boundaries'], peak_width_dates_mask))
    epidemiology_columns['peak_width_heights'] = np.concatenate(
        (epidemiology_columns['peak_width_heights'], peak_width_heights_mask))

    ###Finding dates above threshold
    thresh = np.percentile(ys, THRESHOLD)
    days_above = pd.Series((ys>np.percentile(ys,THRESHOLD)).astype(int))
    waves = [(cat[1], grp.shape[0]) for cat, grp in days_above.groupby([
        days_above.ne(days_above.shift()).cumsum(), days_above])]


    wave_number = 0
    for i,wave in enumerate(waves):
        if wave[0]==0:
            continue
        if wave[1]<MIN_PERIOD:
            days_above.iloc[sum([wave[1] for x in range(i)]):
                            sum([wave[1] for x in range(i)]) + wave[1]] = 0
            continue
        wave_number += 1
        days_above.iloc[sum([waves[x][1] for x in range(i)]):
                        sum([waves[x][1] for x in range(i)]) + wave[1]] = wave_number

    threshold_average_height_mask = np.zeros(len(data['date']))
    threshold_max_height_mask = np.zeros(len(data['date']))

    for i in range(wave_number):
        threshold_average_height_mask[i] = (ys[np.where(days_above == (i+1))]/np.max(ys)).mean()
        threshold_max_height_mask[i] = np.max(ys[np.where(days_above == (i+1))])/np.max(ys)

    ###Upserting data
    epidemiology_columns['threshold_dates'] = np.concatenate((epidemiology_columns['threshold_dates'],
                                                              days_above))
    epidemiology_columns['threshold_average_height'] = np.concatenate((epidemiology_columns['threshold_average_height'],
                                                              threshold_average_height_mask))
    epidemiology_columns['threshold_max_height'] = np.concatenate((epidemiology_columns['threshold_max_height'],
                                                              threshold_max_height_mask))
    if SAVE_PLOTS:
        plt.figure(figsize = (20,7))
        plt.title('New Cases Per Day with Spline Fit for ' + country)
        plt.ylabel('new_per_day')
        plt.xlabel('date')

        ###Plotting new_per_day curves
        plt.plot(data['date'].values, data['new_per_day'].values, label = 'new_per_day')
        plt.plot(data['date'].values, data['new_per_day_ma'].values, label = 'new_per_day_ma')
        plt.plot(data['date'].values, ys, label = 'new_per_day_smooth')
        plt.plot([data['date'].values[i] for i in peak_locations], [ys[i] for i in peak_locations], "X", ms=20)

        ###Plotting peak_widths
        for i in range(len(peak_locations)):
            plt.hlines(y=width_heights[i], xmin=data['date'].values[int(width_left[i])],
                       xmax=data['date'].values[int(width_right[i])],
                       linestyle='dotted', label='width_peak_' + str(i + 1))

        ###Plotting wave_threshold
        plt.hlines(y=thresh, xmin=data['date'].values[0],
                   xmax=data['date'].values[-1], label=str(THRESHOLD) + 'th_percentile',
                   linestyles='dashed', color='red')
        plt.legend()
        plt.savefig(PATH+'epidemiological/'+country+'.png')
        plt.close()


'''
MOBILITY PROCESSING
'''

for country in tqdm(countries, desc = 'Processing Mobility Data'):
    data = mobility[mobility['countrycode'] == country]

    mobility_columns['countrycode'] = np.concatenate((mobility_columns['countrycode'], data['countrycode'].values))
    mobility_columns['country'] = np.concatenate((mobility_columns['country'], data['country'].values))
    mobility_columns['date'] = np.concatenate((mobility_columns['date'], data['date'].values))

    for mobility_type in mobilities:
        x = np.arange(len(data['date']))
        y = data[mobility_type].values
        ys = csaps(x, y, x, smooth=SMOOTH)

        peak_locations = find_peaks(ys,distance=DISTANCE)[0]
        trough_locations = find_peaks(-ys,distance=DISTANCE)[0]
        peak_mask = np.zeros(len(data['date']))
        trough_mask = np.zeros(len(data['date']))
        peak_heights_mask = np.zeros(len(data['date']))
        trough_heights_mask = np.zeros(len(data['date']))

        p_prominences = peak_prominences(ys, peak_locations)[0]
        t_prominences = peak_prominences(-ys, trough_locations)[0]
        p_prominences_mask = np.zeros(len(data['date']))
        t_prominences_mask = np.zeros(len(data['date']))

        p_widths, p_width_heights, p_left, p_right = peak_widths(ys, peak_locations, rel_height = 1.0)
        t_widths, t_width_heights, t_left, t_right = peak_widths(-ys, trough_locations, rel_height = 1.0)
        p_widths_mask = np.zeros(len(data['date']))
        t_widths_mask = np.zeros(len(data['date']))

        for i in range(len(peak_locations)):
            peak_mask[peak_locations[i]] = i + 1
            peak_heights_mask[peak_locations[i]] = ys[peak_locations[i]]
            p_prominences_mask[peak_locations[i]] = p_prominences_mask[i]
            p_widths_mask[peak_locations[i]] = p_widths[i]

        for i in range(len(trough_locations)):
            trough_mask[trough_locations[i]] = i + 1
            trough_heights_mask[trough_locations[i]] = ys[trough_locations[i]]
            t_prominences_mask[trough_locations[i]] = t_prominences_mask[i]
            t_widths_mask[trough_locations[i]] = t_widths[i]

        mobility_columns[mobility_type] = np.concatenate((
            mobility_columns[mobility_type],data[mobility_type].values))
        mobility_columns[mobility_type+'_smooth'] = np.concatenate((
            mobility_columns[mobility_type+'_smooth'], ys))
        mobility_columns[mobility_type+'_peak_dates'] = np.concatenate((
            mobility_columns[mobility_type+'_peak_dates'], peak_mask))
        mobility_columns[mobility_type+'_trough_dates'] = np.concatenate((
            mobility_columns[mobility_type+'_trough_dates'], trough_mask))
        mobility_columns[mobility_type+'_peak_heights'] = np.concatenate((
            mobility_columns[mobility_type+'_peak_heights'], peak_heights_mask))
        mobility_columns[mobility_type+'_trough_heights'] = np.concatenate((
            mobility_columns[mobility_type+'_trough_heights'], trough_heights_mask))
        mobility_columns[mobility_type+'_peak_prominences'] = np.concatenate((
            mobility_columns[mobility_type+'_peak_prominences'], p_prominences_mask))
        mobility_columns[mobility_type+'_trough_prominences'] = np.concatenate((
            mobility_columns[mobility_type+'_trough_prominences'], t_prominences_mask))
        mobility_columns[mobility_type+'_peak_widths'] = np.concatenate((
            mobility_columns[mobility_type+'_peak_widths'], p_widths_mask))
        mobility_columns[mobility_type+'_trough_widths'] = np.concatenate((
            mobility_columns[mobility_type+'_trough_widths'], t_widths_mask))

        if SAVE_PLOTS:
            plt.figure(figsize = (20,7))
            plt.title('Change in' + mobility_type + ' mobility with Spline Fit for ' + country)
            plt.ylabel('change_from_baseline')
            plt.xlabel('date')

            plt.plot(data['date'].values, data[mobility_type], label = 'change_from_baseline')
            plt.plot(data['date'].values, ys, label = 'spline_fit')
            plt.plot([data['date'].values[i] for i in peak_locations], [ys[i] for i in peak_locations], "X", ms=20)
            plt.plot([data['date'].values[i] for i in trough_locations], [ys[i] for i in trough_locations], "X", ms=20)

            for i in range(len(peak_locations)):
                plt.hlines(y=p_width_heights[i], xmin=data['date'].values[int(p_left[i])],
                           xmax=data['date'].values[int(p_right[i])],
                           linestyle='dotted', label='width_peak_' + str(i + 1))
            for i in range(len(trough_locations)):
                plt.hlines(y=t_width_heights[i], xmin=data['date'].values[int(t_left[i])],
                           xmax=data['date'].values[int(t_right[i])],
                           linestyle='dotted', label='width_trough_' + str(i + 1))
            plt.legend()
            plt.savefig(PATH+'mobility/'+mobility_type+'/'+country+'.png')
            plt.close()

'''
GOVERNMENT_RESPONSE PROCESSING
'''

for country in tqdm(countries, desc = 'Processing Government Response Data'):
    data = government_response[government_response['countrycode'] == country]

    government_response_columns['countrycode'] = np.concatenate((
        government_response_columns['countrycode'], np.array([country])))
    government_response_columns['country'] = np.concatenate((
        government_response_columns['country'], np.array([data['country'].iloc[0]])))

    max_si = data['stringency_index'].max()
    max_si_start_date = data[data['stringency_index'] == data['stringency_index'].max()]['date'].iloc[0]
    max_si_end_date = data[data['stringency_index'] == data['stringency_index'].max()]['date'].iloc[-1]
    max_si_duration = (max_si_end_date - max_si_start_date).days
    max_si_currently = data[data['stringency_index'] == data['stringency_index'].max()]['date'].iloc[-1] == \
                       data['date'].iloc[-1]

    government_response_columns['max_si'] = np.concatenate((
        government_response_columns['max_si'], np.array([max_si])))
    government_response_columns['max_si_start_date'] = np.concatenate((
        government_response_columns['max_si_start_date'],
        np.array([max_si_start_date])))
    government_response_columns['max_si_end_date'] = np.concatenate((
        government_response_columns['max_si_end_date'],
        np.array([max_si_end_date])))
    government_response_columns['max_si_duration'] = np.concatenate((
        government_response_columns['max_si_duration'],
        np.array([max_si_duration])))
    government_response_columns['max_si_currently'] = np.concatenate((
        government_response_columns['max_si_currently'],
        np.array([max_si_currently]))).astype(bool)

    peak_locations = find_peaks(data['stringency_index'].values, distance = DISTANCE)[0]
    multiple_peaks = len(peak_locations) > 1
    government_response_columns['multiple_peaks'] = np.concatenate((
        government_response_columns['multiple_peaks'],np.array([multiple_peaks]))).astype(bool)

    high_restrictions = data[
        (data['c1_school_closing']==3) |
        (data['c2_workplace_closing']==3) |
        (data['c3_cancel_public_events'] == 2) |
        (data['c4_restrictions_on_gatherings'] == 4) |
        (data['c5_close_public_transport'] == 2) |
        (data['c6_stay_at_home_requirements'] == 3) |
        (data['c7_restrictions_on_internal_movement'] == 2) |
        (data['c8_international_travel_controls'] == 4)]

    if len(high_restrictions)>0:
        high_restrictions_start_date = high_restrictions['date'].iloc[0]
        high_restrictions_end_date = high_restrictions['date'].iloc[-1]
        high_restrictions_duration = (high_restrictions_end_date - high_restrictions_start_date).days
        high_restrictions_current = high_restrictions['date'].iloc[-1] == data['date'].iloc[-1]

        government_response_columns['high_restrictions_start_date'] = np.concatenate((
            government_response_columns['high_restrictions_start_date'], np.array([high_restrictions_start_date])))
        government_response_columns['high_restrictions_end_date'] = np.concatenate((
            government_response_columns['high_restrictions_end_date'], np.array([high_restrictions_end_date])))
        government_response_columns['high_restrictions_duration'] = np.concatenate((
            government_response_columns['high_restrictions_duration'], np.array([high_restrictions_duration])))
        government_response_columns['high_restrictions_current'] = np.concatenate((
            government_response_columns['high_restrictions_current'], np.array([high_restrictions_current])))
    else:
        government_response_columns['high_restrictions_start_date'] = np.concatenate((
            government_response_columns['high_restrictions_start_date'], np.array([0])))
        government_response_columns['high_restrictions_end_date'] = np.concatenate((
            government_response_columns['high_restrictions_end_date'], np.array([0])))
        government_response_columns['high_restrictions_duration'] = np.concatenate((
            government_response_columns['high_restrictions_duration'], np.array([0])))
        government_response_columns['high_restrictions_current'] = np.concatenate((
            government_response_columns['high_restrictions_current'], np.array([0])))

    for percentile in percentiles:
        government_response_columns[str(percentile) + '_duration'] = np.concatenate((
            government_response_columns[str(percentile) + '_duration'],
            np.array([len(data[data['stringency_index'] > percentile]['date'])])))

    flag_mobility = {"c1_school_closing": "residential",
                     "c2_workplace_closing": "workplace",
                     "c3_cancel_public_events": "residential",
                     "c4_restrictions_on_gatherings": "retail_recreation",
                     "c5_close_public_transport": "transit_stations",
                     "c6_stay_at_home_requirements": "residential",
                     "c7_restrictions_on_internal_movement": "residential",
                     "stringency_index": "transit_stations"}

    for key,values in flag_mobility.items():
        x = data[key].values #Flag values
        x_date = data['date'].values
        y = mobility_columns[values + '_smooth'][np.where(mobility_columns['countrycode'] == country)] #Smooth Mobility
        y_date = mobility_columns['date'][np.where(mobility_columns['countrycode'] == country)]

        if len(x) > len(y):
            x = x[np.isin(x_date,y_date)]
        else:
            y = y[np.isin(y_date,x_date)]

        ## Bryan could you have a check here?
        if values != 'residential':
            government_response_columns[key + '_opt_lag'] = np.append(government_response_columns[key + '_opt_lag'],
                np.argmin([sm.OLS(endog = y, exog = sm.add_constant(shift(x,lag,cval=np.nan)), missing="drop").fit().params[0]
             for lag in range(-OPTIMAL_LAG_TIME,OPTIMAL_LAG_TIME + 1)]) - OPTIMAL_LAG_TIME)
        else:
            government_response_columns[key + '_opt_lag'] = np.append(government_response_columns[key + '_opt_lag'],
                np.argmax([sm.OLS(endog = y, exog = sm.add_constant(shift(x,lag,cval=np.nan)), missing="drop").fit().params[0]
             for lag in range(-OPTIMAL_LAG_TIME,OPTIMAL_LAG_TIME + 1)]) - OPTIMAL_LAG_TIME)

    if multiple_peaks:
        p_widths, p_heights, p_left, p_right = peak_widths(data['stringency_index'].values,
                                                           peak_locations,rel_height=1.0)
        p_prominences = peak_prominences(data['stringency_index'].values, peak_locations)[0]
        peak_midpoint_date = [data['date'].values[i] for i in peak_locations]
        peak_heights = [data['stringency_index'].values[i] for i in peak_locations]
        peak_start_date = [tuple(data['date'].values[i-1]
                               for i in range(peak_locations[j],0,-1)
                               if (data['stringency_index'].values[i]!=data['stringency_index'].values[i-1]))[0]
                         for j in range(len(peak_locations))]
        peak_end_date = [tuple(data['date'].values[i-1]
                               for i in range(peak_locations[j],len(data['date']),1)
                               if (data['stringency_index'].values[i]!=data['stringency_index'].values[i-1]))[0]
                         for j in range(len(peak_locations))]

        government_response_columns['peak_heights'] = np.append(
            government_response_columns['peak_heights'], set(peak_heights))
        government_response_columns['peak_start_date'] = np.append(
            government_response_columns['peak_start_date'], set(peak_start_date))
        government_response_columns['peak_end_date'] = np.append(
            government_response_columns['peak_end_date'], set(peak_end_date))
        government_response_columns['peak_widths'] = np.append(
            government_response_columns['peak_widths'], set(p_widths))
        government_response_columns['peak_prominences'] = np.append(
            government_response_columns['peak_prominences'], set(p_prominences))

    else:
        government_response_columns['peak_heights'] = np.concatenate((
            government_response_columns['peak_heights'], np.array([0])))
        government_response_columns['peak_start_date'] = np.concatenate((
            government_response_columns['peak_start_date'], np.array([0])))
        government_response_columns['peak_end_date'] = np.concatenate((
            government_response_columns['peak_end_date'], np.array([0])))
        government_response_columns['peak_widths'] = np.concatenate((
            government_response_columns['peak_widths'], np.array([0])))
        government_response_columns['peak_prominences'] = np.concatenate((
            government_response_columns['peak_prominences'], np.array([0])))



    if SAVE_PLOTS:
        plt.figure(figsize = (20,7))
        plt.title('Stringency Index against Time for '+country)
        plt.ylabel('stringency_index')
        plt.xlabel('date')

        plt.plot(data['date'].values, data['stringency_index'].values, label = 'stringency_index')
        plt.plot([data['date'].values[i] for i in peak_locations], [data['stringency_index'].values[i]
                                                                    for i in peak_locations], "X", ms=20)
        for percentile in percentiles:
            plt.hlines(y = percentile, xmin = data['date'].values[0], xmax = data['date'].values[-1],
                       linestyles='dashed', label = str(percentile), colors = 'red')
        plt.legend()
        plt.savefig(PATH+'government_response/'+country+'.png')
        plt.close()

'''
CONSOLIDATING TABLES
'''

epidemiology_results = pd.DataFrame.from_dict(epidemiology_columns)
mobility_results = pd.DataFrame.from_dict(mobility_columns)
government_response_results = pd.DataFrame.from_dict(government_response_columns)

EPI = pd.DataFrame(columns = ['COUNTRYCODE','COUNTRY','EPI_CONFIRMED','EPI_NUMBER_PEAKS',''])

"""
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

if SAVE_PLOTS:
    os.makedirs(PATH+'epidemiological/',exist_ok=True)

##MOBILITY TABLE
mobility_columns={
    'countrycode':np.empty(0),
    'country':np.empty(0),
    'date':np.empty(0)
    }

if SAVE_PLOTS:
    os.makedirs(PATH+'mobility/',exist_ok=True)

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

percentiles = [25, 50, 75]
government_response_columns={
    'countrycode': np.empty(0),
    'country': np.empty(0),
    'max_si':np.empty(0),
    'max_si_start_date':np.empty(0),
    'max_si_end_date':np.empty(0),
    'max_si_duration':np.empty(0),
    'max_si_currently':np.empty(0),
    'multiple_peaks':np.empty(0),
    'peak_heights':np.empty(0),
    'peak_start_date':np.empty(0),
    'peak_end_date':np.empty(0),
    'peak_widths':np.empty(0),
    'peak_prominences':np.empty(0),
    'high_restrictions_start_date':np.empty(0),
    'high_restrictions_end_date':np.empty(0),
    'high_restrictions_duration':np.empty(0),
    'high_restrictions_current':np.empty(0),
    'stringency_index_opt_lag':np.empty(0)
}
"""