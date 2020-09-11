'''
HUMAN-EYE PEAK CLASSIFICATION ALGORITHM

Order the peaks from largest to smallest.
Remove peak candidates too close to the end of the time series.
Starting with the largest, loop through to identify the first genuine peak:
    Does there exist a value after the peak candidate that is <=70% of the peak value?
    Is the peak candidate >=50% of the global maximum value?
    If yes, label the peak candidate genuine.
    If no, go to next candidate.
    
Thereafter, loop through the peak candidates in descending order to identify the remaining genuine peaks:
    Is the peak candidate at least 50% of the value of the tallest genuine peak?
    Does there exist a value after the peak candidate that is <=70% of the peak value?
    Does there exist a value between the peak candidate and the nearest genuine peak (on either side) <=50% of the peak candidate value?
    If yes, label the peak candidate genuine.
    If no, go to next candidate.
'''
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
import datetime
import seaborn as sns
import psycopg2
import warnings
from csaps import csaps
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

#%% File directories
# Source path to get manual peak labels
SOURCE_PATH = ""
# Path to save plots of country time series and csv
SAVE_PATH = ""

SAVE_PLOTS = True
SAVE_CSV = True

if SAVE_PLOTS or SAVE_CSV:
    os.makedirs(SAVE_PATH + 'Peak Classification/', exist_ok=True)
    
#%% Parameters
SMOOTH = 0.001
DISTANCE = 21
PROMINENCE_THRESHOLD = 5
ABSOLUTE_T0_THRESHOLD = 1000
POP_RELATIVE_T0_THRESHOLD = 5 #per million people

#%% Get epidemiology time series data
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

# GET COUNTRY POPULATIONS (2011 - 2019 est.)
indicator_code = 'SP.POP.TOTL'
sql_command = """SELECT countrycode, value, year FROM world_bank WHERE 
adm_area_1 IS NULL AND indicator_code = %(indicator_code)s"""
wb_statistics = pd.read_sql(sql_command, conn, params={'indicator_code': indicator_code})
assert len(wb_statistics) == len(wb_statistics['countrycode'].unique())
wb_statistics = wb_statistics.sort_values(by=['countrycode', 'year'], ascending=[True, False]).reset_index(drop=True)

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
    'days_since_t0_pop':np.empty(0)
}

'''
EPIDEMIOLOGY TIME SERIES PROCESSING
'''

countries = np.sort(epidemiology['countrycode'].unique())
for country in tqdm(countries, desc='Processing Epidemiological Time Series Data'):
    data = epidemiology[epidemiology['countrycode'] == country]
    
    x = np.arange(len(data['date']))
    y = data['new_per_day'].values
    ys = csaps(x, y, x, smooth=SMOOTH)
    z = data['dead_per_day'].values
    zs = csaps(x, z, x, smooth=SMOOTH)

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
    epidemiology_series['days_since_t0_pop'] = np.concatenate(
        (epidemiology_series['days_since_t0_pop'], days_since_t0_relative))

epidemiology_series = pd.DataFrame.from_dict(epidemiology_series)
countries = epidemiology_series['countrycode'].unique()


#%% Get manual labels
# GET COUNTRY LABELS
class_dictionary = {
    'EPI_ENTERING_FIRST': 1,
    'EPI_PAST_FIRST': 2,
    'EPI_ENTERING_SECOND': 3,
    'EPI_PAST_SECOND': 4}
labelled_columns = pd.read_csv(SOURCE_PATH + 'peak_labels.csv')

#%% Parameters for classification
x = 7  # Number of days minimum from end of time series
y = 0.3 # For first peak candidate: % of max value of time series
z = 0.7 # For first peak candidate: there must exist a value after the peak <=z% of the peak value
a = 0.3 # For following peak candidates: % of value of largest genuine peak
b = 0.7 # For following peak candidates: there must exist a value after the peak <=b% of the peak value
d = 0.7 # For following peak candidates: there must exist a trough between any 2 peaks <=d% the value of the smaller of the 2 peaks
e = 50  # Absolute height threshold for genuine peak
f = 0.5 # For a country to be entering a new wave, there has to exist a value after the minima following the last genuine peak that is >=f% of the smallest genuine peak
g = 50  # Absolute height threshold for a country with no genuine peaks to be entering first wave


#%% Initialize dataframe for classification
peaks = pd.DataFrame(columns=["countrycode","number","date","value","genuine","manual_genuine"])
classes = pd.DataFrame(columns=["countrycode","class","manual_class"])
gov_max_peaks = 4

#%% Classify peaks for each country
for c in tqdm(countries, desc="Classifying peaks"):
    # Get peak indices
    peak_characteristics = find_peaks(
        epidemiology_series[epidemiology_series['countrycode']==c]['new_per_day_smooth'].values,
        prominence=PROMINENCE_THRESHOLD, distance=DISTANCE)
    # Initialize peaks dataframe
    c_peaks = pd.DataFrame(columns=["series_index","countrycode","number","date","value","genuine","manual_genuine"])
    c_peaks['series_index'] = [i for i in peak_characteristics[0]]
    c_peaks['date'] = [epidemiology_series[epidemiology_series['countrycode']==c]['date'].values[i] for i in peak_characteristics[0]]
    c_peaks['value'] = [epidemiology_series[epidemiology_series['countrycode']==c]['new_per_day_smooth'].values[i] for i in peak_characteristics[0]]
    c_peaks['countrycode'] = c
    c_peaks['number'] = [n for n in range(1,len(peak_characteristics[0])+1)]
    # Get time series for country
    c_series = epidemiology_series.loc[epidemiology_series["countrycode"]==c,["date","new_per_day_smooth"]]
    if len(c_peaks) > 0:
        # Remove all peak candidates <=x days before the end of the time series
        # Also removes all N/A peaks
        end_date = c_series["date"].iloc[-1]
        c_peaks = c_peaks.loc[c_peaks["date"]<=end_date-datetime.timedelta(days=x),:]
        # Remove all peak candidates <e absolute height
        c_peaks = c_peaks.loc[c_peaks["value"]>=e,:]
        if len(c_peaks) > 0:
            # Sort by peak value
            c_peaks.sort_values(by='value',ascending=False,inplace=True)
            c_peaks.reset_index(drop=True,inplace=True)
            # Get max value of time series
            max_value = max(c_series['new_per_day_smooth'])
            # Loop through each peak to find the first genuine peak
            for i in c_peaks.index:
                #Is the peak candidate >=y% of the global maximum value?
                if c_peaks.loc[i,"value"] >= y*max_value:
                #Does there exist a value after the peak candidate that is <=z% of the peak value?
                    if len(c_series.loc[(c_series['date']>c_peaks.loc[i,"date"]) & \
                                        (c_series['new_per_day_smooth']<=z*c_peaks.loc[i,"value"])]) > 0:
                        # If yes, mark the peak as genuine.
                        c_peaks.loc[i,"genuine"]=True
                        # Stop the loop as the first genuine peak has been found
                        break
                    else:
                        # If no, mark the peak as non-genuine
                        c_peaks.loc[i,"genuine"]=False
                        # Continue to search for the first genuine peak
                else:
                    c_peaks['genuine']=False
                    break
            
            # Loop through each peak to find the remaining genuine peaks
            for i in c_peaks.index:
                # Skip those peak candidates that have already been labelled
                if c_peaks.loc[i,"genuine"]==True or c_peaks.loc[i,"genuine"]==False:
                    continue
                else:
                    #Is the peak candidate at least a% of the value of the tallest genuine peak?
                    if c_peaks.loc[i,"value"] >= a*max(c_peaks.loc[c_peaks["genuine"]==True,"value"]):
                        #Does there exist a value after the peak candidate that is <=b% of the peak value?
                        if len(c_series.loc[(c_series['date']>c_peaks.loc[i,"date"]) & \
                                            (c_series['new_per_day_smooth']<=b*c_peaks.loc[i,"value"])]) > 0:
                            # Does there exist a value between the peak candidate and the nearest genuine peak (on either side) <=50% of the peak candidate value?
                            genuine_dates = []
                            genuine_dates = list(c_peaks.loc[c_peaks["genuine"]==True,'date'])
                            genuine_dates.append(c_peaks.loc[i,"date"])
                            genuine_dates.sort()
                            candidate_index = genuine_dates.index(c_peaks.loc[i,"date"])
                            if candidate_index >=1:
                                left_date = genuine_dates[candidate_index-1]
                            else:
                                left_date = 0
                            if candidate_index+1 <=len(genuine_dates)-1:
                                right_date = genuine_dates[candidate_index+1]
                            else:
                                right_date = 0
                            # Check left side and right side:
                            if ((left_date==0) or (len(c_series.loc[(c_series['date']<c_peaks.loc[i,"date"]) & \
                                                (c_series['date']>left_date) & \
                                                (c_series['new_per_day_smooth']<=d*c_peaks.loc[i,"value"])]) > 0)) \
                            and ((right_date==0) or (len(c_series.loc[(c_series['date']>c_peaks.loc[i,"date"]) & \
                                                (c_series['date']<right_date) & \
                                                (c_series['new_per_day_smooth']<=d*c_peaks.loc[i,"value"])]) > 0)):
                                # Mark the peak as genuine
                                c_peaks.loc[i,"genuine"]=True
                            else:
                                # Mark the peak as non-genuine
                                c_peaks.loc[i,"genuine"]=False
                        else:
                            # Mark the peak as non-genuine
                            c_peaks.loc[i,"genuine"]=False
                    else:
                        # Mark the peak as non-genuine
                        c_peaks.loc[i,"genuine"]=False
    
    # Get actual manual labels
    for n in range(1,gov_max_peaks+1):
        c_peaks.loc[(c_peaks["number"]==n),"manual_genuine"]=labelled_columns.loc[labelled_columns["COUNTRYCODE"]==c,"EPI_PEAK_"+str(n)+"_GENUINE"].values[0]
    
    # Append labels to dataframe
    peaks = peaks.append(c_peaks,ignore_index=True) 
    
    # Plot time series to see the labelling
    if SAVE_PLOTS == True:
        plt.clf()
        plt.close("all")
        fig = sns.lineplot(data=c_series, x="date",y="new_per_day_smooth")
        for i in c_peaks.index:
            plt.text(x=c_peaks.loc[i,"date"],y=1.05*c_peaks.loc[i,"value"], s="Classification: "+str(c_peaks.loc[i,"genuine"]))
            plt.text(x=c_peaks.loc[i,"date"],y=1.15*c_peaks.loc[i,"value"], s="Manual labels: "+str(c_peaks.loc[i,"manual_genuine"]))
        plt.title('New Cases Per Day Smoothed for ' + c)
        ymin, ymax = fig.get_ylim()
        fig.set_ylim(ymin,1.2*ymax)
        plt.savefig(SAVE_PATH + "Peak Classification/" + c + '.png')

    # Assign country to class based on number of genuine peaks
    # If n=0, CLASS=0, if n=1, CLASS=2, if n=2, CLASS=4
    c_class = 2*len(c_peaks.loc[c_peaks["genuine"]==True])
    # Increase the class by 1 if the country is entering a new peak
    if c_class > 0:
        # Entering a new peak iff: after the minima following the last genuine peak, there is a value >=f% of the smallest peak value
        try:
            last_peak_date = max(c_peaks.loc[c_peaks["genuine"]==True,"date"])
            trough_date = c_series.loc[c_series["new_per_day_smooth"]==min(c_series.loc[c_series["date"]>last_peak_date,"new_per_day_smooth"]),"date"].values[0]
            max_after_trough = max(c_series.loc[c_series["date"]>trough_date,"new_per_day_smooth"])
            if max_after_trough >= f*min(c_peaks.loc[c_peaks["genuine"]==True,"value"]):
                c_class = c_class + 1
        except ValueError: # if error due to empty sequence, then move to next country
            pass
    else: # if 0 genuine peaks, mark as entering first wave if max height passes an absolute threshold g
        if max(c_series["new_per_day_smooth"]) >= g:
            c_class = c_class + 1
    # Get manual labels
    manual_class = 0 if np.sum(labelled_columns[labelled_columns['COUNTRYCODE']==c][[
        'EPI_ENTERING_FIRST', 'EPI_PAST_FIRST', 'EPI_ENTERING_SECOND', 'EPI_PAST_SECOND']].values) == 0 else \
        class_dictionary[labelled_columns[labelled_columns['COUNTRYCODE']==c][[
        'EPI_ENTERING_FIRST', 'EPI_PAST_FIRST', 'EPI_ENTERING_SECOND', 'EPI_PAST_SECOND']].idxmax(axis=1).values[0]]

    # Append class to dataframe
    classes = classes.append(pd.DataFrame([[c,c_class,manual_class]],columns=["countrycode","class","manual_class"]),ignore_index=True)

#%% Get accuracy
true_positive = len(peaks.loc[(peaks["genuine"]==True)&(peaks["manual_genuine"]==True)])
false_positve = len(peaks.loc[(peaks["genuine"]==True)&(peaks["manual_genuine"]==False)])
true_negative = len(peaks.loc[(peaks["genuine"]==False)&(peaks["manual_genuine"]==False)])
false_negative = len(peaks.loc[(peaks["genuine"]==False)&(peaks["manual_genuine"]==True)])

print("True positive: "+str(true_positive))
print("False positive: "+str(false_positve))
print("True negative: "+str(true_negative))
print("False negative: "+str(false_negative))
print("Accuracy: "+str((true_positive+true_negative)/(true_positive+true_negative+false_positve+false_negative)))


#%% Save csv
if SAVE_CSV:
    peaks.merge(classes,on="countrycode").to_csv(SAVE_PATH + "/Peak Classification/peak_classification.csv")
