import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from tqdm import tqdm
import datetime
from datetime import date

#%%
SAVE_PLOTS = True
READ_CSV = True
PATH = ''

#%% Read CSVs
if READ_CSV:
    FINAL = pd.read_csv(PATH +'master.csv',index_col = 0, parse_dates=['EPI_GENUINE_PEAK_1_DATE','GOV_PEAK_1_START_DATE','GOV_PEAK_1_END_DATE','T0'])
    epidemiology_results = pd.read_csv(PATH +'epidemiology_results.csv',index_col = 0, parse_dates=['date'])
    mobility_results = pd.read_csv(PATH +'mobility_results.csv',index_col = 0, parse_dates=['date'])
    government_response_results = pd.read_csv(PATH +'government_response_results.csv',index_col = 0, parse_dates=['high_restrictions_start_date','high_restrictions_end_date'])
    government_response = pd.read_csv(PATH +'government_response.csv',index_col = 0, parse_dates=['date'])

#%% Labelling country classes

CLASS_LABELS_DICTIONARY = {
    'Other' : 0,
    'Entering_First_Wave' : 1,
    'Past_First_Wave' : 2,
    'Entering_Second_Wave' : 3,
    'Past_Second_Wave' : 4
}

for c in CLASS_LABELS_DICTIONARY:
    FINAL.loc[FINAL['CLASS']==CLASS_LABELS_DICTIONARY[c],'CLASS_LABEL'] = c
    
#%% Create directories to save plots
if SAVE_PLOTS:
    for c in CLASS_LABELS_DICTIONARY:
        class_label = str(int(CLASS_LABELS_DICTIONARY[c])) + '_' + c
        os.makedirs(PATH+'time_series_plots/'+class_label+'/',exist_ok=True)
        os.makedirs(PATH+'dist_plots/',exist_ok=True)


#%% Scatterplot of duration of first wave vs. duration of stringency
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
if SAVE_PLOTS:
    g.figure.savefig(PATH + 'duration_scatter.png')


#%% Time series EPI - GOV (number of new cases and government lockdown flags for each country)

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

#%% Consolidate data for SI and mobility plots

# Merge tables together for plotting
data = government_response[['date','stringency_index','countrycode']]
data = data.merge(FINAL[['CLASS_LABEL','COUNTRYCODE']],left_on='countrycode', right_on='COUNTRYCODE', how='left')
data = data.merge(mobility_results[['countrycode','date','residential','transit_stations','workplace']], on=['countrycode','date'], how='left')
data = data.merge(epidemiology_results[['countrycode','date','new_per_day']], on=['countrycode','date'], how='left')

data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave']),'CLASS_COARSE'] = 'First_Wave'
data.loc[data['CLASS_LABEL'].isin(['Entering_Second_Wave','Past_Second_Wave']),'CLASS_COARSE'] = 'Second_Wave'

# Define t as days since T0
for c in data['countrycode'].unique():
    try:
        T0 = FINAL.loc[FINAL['COUNTRYCODE']==c,'T0'].values[0]
        if isinstance(T0, str):
            data.loc[data['countrycode']==c,'t'] = [a.days for a in (data.loc[data['countrycode']==c,'date'] - datetime.datetime.strptime(T0, '%Y-%m-%d').date())]
        elif isinstance(T0, datetime.date) or isinstance(T0, np.datetime64):
            data.loc[data['countrycode']==c,'t'] = [a.days for a in (data.loc[data['countrycode']==c,'date'] - T0)]
        else:
            data.loc[data['countrycode']==c,'t'] = np.nan
    except IndexError:
        data.loc[data['countrycode']==c,'t'] = np.nan

#%% Plot of average stringency index over time for each class
# SI plots: by date
data_plot = data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave']),['CLASS_LABEL','date','stringency_index']]
data_plot.dropna(how='any', inplace=True)
plt.clf()
plt.figure(figsize = (15,7))
sns.set_style("darkgrid")
g = sns.lineplot(x = 'date', y ='stringency_index', data=data_plot, hue = 'CLASS_LABEL',
             hue_order = ['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave'])
plt.title('Government Stringency Over Time for Each Country Cluster')
plt.xlabel('Date')
plt.ylabel('Stringency Index')
if SAVE_PLOTS:
    g.figure.savefig(PATH + 'SI_per_class.png')

# SI plots: by days since T0
data_plot = data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave']),['CLASS_LABEL','t','stringency_index']]
data_plot.dropna(how='any', inplace=True)
plt.clf()
plt.figure(figsize = (15,7))
sns.set_style("darkgrid")
g = sns.lineplot(x = 't', y ='stringency_index', data=data_plot, hue = 'CLASS_LABEL',
             hue_order = ['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave'])
g.set(xlim=(-75, 145))
plt.title('Government Stringency Over Time for Each Country Cluster')
plt.xlabel('Days Since T0 (First Day of 50 Cumulative Cases)')
plt.ylabel('Stringency Index')
if SAVE_PLOTS:
    g.figure.savefig(PATH + 'SI_per_class_t.png')

# SI plots: by days since T0, first vs. second wave countries
data_plot = data.loc[data['CLASS_COARSE'].isin(['First_Wave','Second_Wave']),['CLASS_COARSE','t','stringency_index']]
data_plot.dropna(how='any', inplace=True)
plt.clf()
plt.figure(figsize = (15,7))
sns.set_style("darkgrid")
g = sns.lineplot(x = 't', y ='stringency_index', data=data_plot, hue = 'CLASS_COARSE',
             hue_order = ['First_Wave','Second_Wave'])
g.set(xlim=(-75, 145))
plt.title('Government Stringency Over Time for First and Second Wave Countries')
plt.xlabel('Days Since T0 (First Day of 50 Cumulative Cases)')
plt.ylabel('Stringency Index')
if SAVE_PLOTS:
    g.figure.savefig(PATH + 'SI_per_class_coarse_t.png')

#%% Plot of average mobility over time for each class 
# Mobility plots: by days since T0
if SAVE_PLOTS:
    for m in ['residential','transit_stations','workplace']:
        data_plot = data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave']),['CLASS_LABEL','t',m]]
        data_plot.dropna(how='any', inplace=True)
        plt.close('all')
        plt.clf()
        plt.figure(figsize = (15,7))
        sns.set_style("darkgrid")
        g = sns.lineplot(x = 't', y =m, data=data_plot, hue = 'CLASS_LABEL',
                     hue_order = ['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave'])
        g.set(xlim=(-30, 145))
        plt.title(m + ' Mobility Over Time for Each Country Cluster')
        plt.xlabel('Days Since T0 (First Day of 50 Cumulative Cases)')
        plt.ylabel(m)
        g.figure.savefig(PATH + 'MOB_' + m + '_per_class_t.png')

#%% Scatterplot of response time vs. number of cumulative cases
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
plt.title('Government Response Time Against Number of Confirmed Cases')
plt.xlabel('Response Time: Days from T0 to Peak Date of Stringency Index')
plt.ylabel('Cumulative Number of Confirmed Cases per 10,000')
if SAVE_PLOTS:
    g.figure.savefig(PATH + 'response_scatter_2.png')


#%% Median SI and new cases over time
data_plot = data.loc[data['CLASS_COARSE'].isin(['First_Wave','Second_Wave']),['CLASS_COARSE','t','new_per_day','stringency_index']]
data_plot.dropna(how='any', inplace=True)
data_plot_mean = data.groupby(['CLASS_COARSE','t'],as_index=False).agg({'new_per_day':'mean','stringency_index':'mean'})
data_plot_median = data.groupby(['CLASS_COARSE','t'],as_index=False).agg({'new_per_day':'median','stringency_index':'median'})


plt.close('all')
plt.clf()
plt.style.use('seaborn')
g, ax1 =plt.subplots(figsize=(20,7))
# EPI: New Cases per Day
#ax1.plot(data_plot_median.loc[data_plot_median['CLASS_COARSE']=='First_Wave','t'].values, data_plot_median.loc[data_plot_median['CLASS_COARSE']=='First_Wave','new_per_day'].values,
#         label='New Cases per Day: First Wave Countries', color='b')
ax1.plot(data_plot_median.loc[data_plot_median['CLASS_COARSE']=='Second_Wave','t'].values, data_plot_median.loc[data_plot_median['CLASS_COARSE']=='Second_Wave','new_per_day'].values,
         label='New Cases per Day: Second Wave Countries', color='g')
# GOV: Stringency Index
ax2 = ax1.twinx()
#ax2.plot(data_plot_median.loc[data_plot_median['CLASS_COARSE']=='First_Wave','t'].values, data_plot_median.loc[data_plot_median['CLASS_COARSE']=='First_Wave','stringency_index'].values,
#         label='Stringency Index: First Wave Countries', color='b')
ax2.plot(data_plot_median.loc[data_plot_median['CLASS_COARSE']=='Second_Wave','t'].values, data_plot_median.loc[data_plot_median['CLASS_COARSE']=='Second_Wave','stringency_index'].values,
         label='Stringency Index: Second Wave Countries', color='r')
# Labels
ax1.set_ylabel('New Cases per Day')
ax1.set_xlabel('Date')
ax2.set_ylabel('Stringency Index')
ax2.grid(None)
plt.title('New Cases Per Day Against Government Response')
ax1.figure.legend()
#%% Correlation matrix

final_corr = FINAL.corr()
corr_list = final_corr.dropna(how='all').index
final_corr = FINAL[corr_list].corr()
if SAVE_PLOTS:
    final_corr.to_csv(PATH + 'final_corr.csv')


#%% Testing: Distribution plots

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

#%% Testing: distribution plots of each variable
FINAL_working = FINAL
for c in FINAL_working.columns:
    if 'DATE' in c:
        try:
            pd.to_datetime(FINAL_working[c],format='%Y-%m-%d')
        except:
            print('Column not converted: ' + c)

col_list = FINAL_working.loc[:,(FINAL_working.dtypes==float)|(FINAL_working.dtypes==datetime)].columns

class_list = ['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave']

if SAVE_PLOTS:
    for m in tqdm(col_list):
        try:
            plt.clf()
            plt.figure(figsize = (10,7))
            for c in class_list:
                sns.distplot(FINAL.loc[FINAL['CLASS_LABEL']==c,m].dropna(), hist=False, 
                             label=c + ' N=' + str(len(FINAL.loc[FINAL['CLASS_LABEL']==c,m].dropna())))
            plt.title('Distribution by Class: ' + m)
            plt.xlabel(m)
            plt.ylabel('Density')
            plt.savefig(PATH + 'dist_plots/' + m + '.png')
            plt.close('all')
        except:
            print('Column not plotted: ' + m)

