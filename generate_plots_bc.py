import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import warnings
from tqdm import tqdm
import datetime
from datetime import date

#%%
SAVE_PLOTS = True
READ_CSV = True
PATH = 'C:/Users/bryan/OneDrive/Desktop/Epidemetrics/'

# Read CSVs
if READ_CSV:
    FINAL = pd.read_csv(PATH +'master.csv',index_col = 0, parse_dates=['EPI_GENUINE_PEAK_1_DATE','GOV_PEAK_1_START_DATE','GOV_PEAK_1_END_DATE','T0'])
    epidemiology_results = pd.read_csv(PATH +'epidemiology_results.csv',index_col = 0, parse_dates=['date'])
    mobility_results = pd.read_csv(PATH +'mobility_results.csv',index_col = 0, parse_dates=['date'])
    government_response_results = pd.read_csv(PATH +'government_response_results.csv',index_col = 0, parse_dates=['high_restrictions_start_date','high_restrictions_end_date'])


#%% labelling classes

CLASS_LABELS_DICTIONARY = {
    'Other' : 0,
    'Entering_First_Wave' : 1,
    'Past_First_Wave' : 2,
    'Entering_Second_Wave' : 3,
    'Past_Second_Wave' : 4
}

for c in CLASS_LABELS_DICTIONARY:
    FINAL.loc[FINAL['CLASS']==CLASS_LABELS_DICTIONARY[c],'CLASS_LABEL'] = c
    
# Create directories to save plots
if SAVE_PLOTS:
    for c in CLASS_LABELS_DICTIONARY:
        class_label = str(int(CLASS_LABELS_DICTIONARY[c])) + '_' + c
        os.makedirs(PATH+'time_series_plots/'+class_label+'/',exist_ok=True)

#%% scatterplot of duration of first wave vs. duration of stringency
data=FINAL.loc[FINAL['CLASS']!=1,['GOV_PEAK_1_WIDTH','EPI_GENUINE_PEAK_1_WIDTH','CLASS_LABEL']].dropna(how='any')

plt.clf()
plt.figure(figsize = (15,10))
sns.set_style("darkgrid")
g = sns.scatterplot(x='GOV_PEAK_1_WIDTH',y='EPI_GENUINE_PEAK_1_WIDTH',data=data,
                    hue='CLASS_LABEL',palette=sns.color_palette("muted", data.CLASS_LABEL.nunique()),
                    hue_order = ['Past First Wave','Entering Second Wave','Past Second Wave','Other'],
                    style='CLASS_LABEL', s=50)
#g.set(xlim=(date(2020,1,1), date(2020,8,1)))
#g.set(ylim=(date(2020,1,1), date(2020,8,1)))
plt.title('Duration of First Wave Against Duration of Stringency')
plt.xlabel('Duration of First Wave of Government Restrictions (Days)')
plt.ylabel('Duration of First Wave of Covid-19 Cases (Days)')
g.figure.savefig(PATH + 'duration_scatter.png')


#%% time series EPI - GOV

countries = np.sort(FINAL['COUNTRYCODE'].unique())

if SAVE_PLOTS:
    for country in tqdm(countries, desc = 'Generating Plots'):
        plt.close('all')
        plt.clf()
        plt.style.use('seaborn')
        g, ax1 =plt.subplots(figsize=(20,7))
        country_name = FINAL.loc[FINAL['COUNTRYCODE']==country,'COUNTRY'].values[0]
        class_label = str(int(FINAL.loc[FINAL['COUNTRYCODE']==country,'CLASS'].values[0])) \
                      + '_' + FINAL.loc[FINAL['COUNTRYCODE']==country,'CLASS_LABEL'].values[0]
        # GOV: High Restrictions
        try:
            start_date = np.datetime64(government_response_results.loc[government_response_results['countrycode']==country,'high_restrictions_start_date'].values[0])
            end_date = np.datetime64(government_response_results.loc[government_response_results['countrycode']==country,'high_restrictions_end_date'].values[0])
            plt.axvline(start_date, linestyle='dotted', color='red', label='Gov High Restrictions')
            plt.axvline(end_date, linestyle='dotted', color='red')
            ax1.axvspan(start_date, end_date, alpha=0.1, color='red')
        except:
            print('No high restrictions for ' + country_name)
        # GOV: Stringency Index Peak
        start_date = FINAL.loc[FINAL['COUNTRYCODE']==country,'GOV_PEAK_1_START_DATE'].values[0]
        end_date = FINAL.loc[FINAL['COUNTRYCODE']==country,'GOV_PEAK_1_END_DATE'].values[0]
        plt.axvline(start_date, linestyle='dashed', color='red', label='Gov Peak Stringency Index')
        plt.axvline(end_date, linestyle='dashed', color='red')
        ax1.axvspan(start_date, end_date, alpha=0.1, color='red')
        # EPI: New Cases per Day
        data = epidemiology_results.loc[epidemiology_results['countrycode']==country,:]
        ax1.plot(data['date'].values, data['new_per_day'].values, label='New Cases per Day')
        ax1.plot(data['date'].values, data['new_per_day_smooth'].values, label='New Cases per Day (Smoothed)', 
                 color='black', linestyle='dashed')
        # GOV: Stringency Index
        ax2 = ax1.twinx()
        data = government_response.loc[government_response['countrycode']==country,:]
        ax2.plot(data['date'].values, data['stringency_index'].values, label='Stringency Index', 
                 color='red')
        # EPI: Peak
        '''
        peak_date = FINAL.loc[FINAL['COUNTRYCODE']==country,'EPI_GENUINE_PEAK_1_DATE'].values[0]
        peak_value = FINAL.loc[FINAL['COUNTRYCODE']==country,'EPI_GENUINE_PEAK_1_VALUE'].values[0]
        ax2.plot(peak_date, peak_value, "X", ms=10, color='r', label='Epi Peak')
        '''
        # Labels
        ax1.set_ylabel('New Cases per Day')
        ax1.set_xlabel('Date')
        ax2.set_ylabel('Stringency Index')
        ax2.grid(None)
        plt.title('New Cases Per Day Against Government Response for ' + country_name)
        ax1.figure.legend()
        plt.savefig(PATH+'time_series_plots/'+class_label+'/'+country+'.png')

#%% plot of average stringency index for each class

data = government_response[['date','stringency_index','countrycode']]
data = data.merge(FINAL[['CLASS_LABEL','COUNTRYCODE']],left_on='countrycode', right_on='COUNTRYCODE', how='left')

for c in data['countrycode'].unique():
    try:
        T0 = FINAL.loc[FINAL['COUNTRYCODE']==c,'T0'].values[0]
        if isinstance(T0, str):
            data.loc[data['countrycode']==c,'t'] = [a.days for a in (data.loc[data['countrycode']==c,'date'] - datetime.datetime.strptime(T0, '%Y-%m-%d').date())]
        else:
            data.loc[data['countrycode']==c,'t'] = np.nan
    except IndexError:
        data.loc[data['countrycode']==c,'t'] = np.nan

data = data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave']),['CLASS_LABEL','date','stringency_index','t']]
data.dropna(how='any', inplace=True)

# SI plots: by date
plt.clf()
plt.figure(figsize = (20,7))
sns.set_style("seaborn")
g = sns.lineplot(x = 'date', y ='stringency_index', data=data, hue = 'CLASS_LABEL',
             hue_order = ['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave'])
plt.title('Government Stringency for Each Country Cluster')
plt.xlabel('Date')
plt.ylabel('Stringency Index')
g.figure.savefig(PATH + 'SI_per_class.png')

# SI plots: by days since T0
plt.clf()
plt.figure(figsize = (20,7))
sns.set_style("darkgrid")
g = sns.lineplot(x = 't', y ='stringency_index', data=data, hue = 'CLASS_LABEL',
             hue_order = ['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave'])
plt.title('Government Stringency for Each Country Cluster')
plt.xlabel('Days Since T0 (First Day of 10 New Cases (Smoothed))')
plt.ylabel('Stringency Index')
g.figure.savefig(PATH + 'SI_per_class_t.png')

#%% distribution plots

plt.clf()
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Entering_First_Wave','GOV_MAX_SI'], hist=False, label='Entering_First_Wave')
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Past_First_Wave','GOV_MAX_SI'], hist=False, label='Past_First_Wave')
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Entering_Second_Wave','GOV_MAX_SI'], hist=False, label='Entering_Second_Wave')

plt.clf()
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Entering_First_Wave','GOV_MAX_SI_DAYS_FROM_T0'], hist=False, label='Entering_First_Wave')
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Past_First_Wave','GOV_MAX_SI_DAYS_FROM_T0'], hist=False, label='Past_First_Wave')
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Entering_Second_Wave','GOV_MAX_SI_DAYS_FROM_T0'], hist=False, label='Entering_Second_Wave')

plt.clf()
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Entering_First_Wave','GOV_PEAK_1_WIDTH'], hist=False, label='Entering_First_Wave')
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Past_First_Wave','GOV_PEAK_1_WIDTH'], hist=False, label='Past_First_Wave')
sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']=='Entering_Second_Wave','GOV_PEAK_1_WIDTH'], hist=False, label='Entering_Second_Wave')


#%% scatterplot of response time vs. SI (response intensity)
data=FINAL.loc[FINAL['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave','Past_Second_Wave']) \
               ,['GOV_MAX_SI_DAYS_FROM_T0','GOV_MAX_SI','CLASS_LABEL','T0','POPULATION','EPI_CONFIRMED','CFR',]].dropna(how='any')

data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave']),'CLASS_COARSE'] = 'First_Wave'
data.loc[data['CLASS_LABEL'].isin(['Entering_Second_Wave','Past_Second_Wave']),'CLASS_COARSE'] = 'Second_Wave'

data['EPI_CONFIRMED_PER_10K'] = 10000*data['EPI_CONFIRMED']/data['POPULATION'] 
data['EPI_LOG_CONFIRMED_PC'] = np.log(data['EPI_CONFIRMED']/data['POPULATION'])
data['EPI_LOG_CONFIRMED'] = np.log(data['EPI_CONFIRMED'])

plt.close('all')
plt.clf()
plt.figure(figsize = (12,8))
sns.set_style("darkgrid")
g = sns.scatterplot(x='GOV_MAX_SI_DAYS_FROM_T0',y='EPI_CONFIRMED_PER_10K',data=data,
                    hue='CLASS_COARSE',palette=sns.color_palette("muted", data.CLASS_COARSE.nunique()),
                    hue_order = ['First_Wave','Second_Wave'],
                    style='CLASS_COARSE', s=50)
#g.set(xlim=(date(2020,1,1), date(2020,8,1)))
#g.set(ylim=(date(2020,1,1), date(2020,8,1)))
plt.title('Government Response Time Against Number of Confirmed Cases')
plt.xlabel('Response Time: Days from T0 to Peak Date of Stringency Index')
plt.ylabel('Cumulative Number of Confirmed Cases per 10,000')
g.figure.savefig(PATH + 'response_scatter_2.png')




