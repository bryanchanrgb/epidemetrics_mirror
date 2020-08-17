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
READ_CSV = False
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

#%% Consolidate data for SI and mobility plots

# Merge tables together for plotting
data = government_response[['date','stringency_index','countrycode']]
data = data.merge(FINAL[['CLASS','CLASS_LABEL','COUNTRYCODE','T0','POPULATION','GOV_C6_RAISED_DATE','GOV_C6_LOWERED_DATE','GOV_C6_RAISED_AGAIN_DATE','EPI_GENUINE_PEAK_1_DATE','EPI_GENUINE_PEAK_2_DATE']],left_on='countrycode', right_on='COUNTRYCODE', how='left')
data = data.merge(mobility_results[['countrycode','date','residential','residential_smooth','transit_stations','workplace']], on=['countrycode','date'], how='left')
data = data.merge(epidemiology_results[['countrycode','date','confirmed','new_per_day','new_per_day_ma','new_per_day_smooth']], on=['countrycode','date'], how='left')

data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave']),'CLASS_COARSE'] = 'First_Wave'
data.loc[data['CLASS_LABEL'].isin(['Entering_Second_Wave','Past_Second_Wave']),'CLASS_COARSE'] = 'Second_Wave'

data['new_per_day_per10k'] = 10000*data['new_per_day']/data['POPULATION']
data['new_per_day_ma_per10k'] = 10000*data['new_per_day_ma']/data['POPULATION']
data['new_per_day_smooth_per10k'] = 10000*data['new_per_day_smooth']/data['POPULATION']

# Define t as days since T0
for c in tqdm(data['countrycode'].unique(), desc='Defining T0'):
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
        
# For countries in second wave: Define s as days since S0 (start of second wave)
# Define S0 as the first day of 50 cases from the local minimum in the region between the first and second peaks
for c in tqdm(data['countrycode'].unique(), desc='Defining S0'):
    if FINAL.loc[FINAL['COUNTRYCODE']==c,'CLASS'].values[0] >=3:
        try:
            peak_1_date = FINAL.loc[FINAL['COUNTRYCODE']==c,'EPI_GENUINE_PEAK_1_DATE'].values[0]
            peak_2_date = FINAL.loc[FINAL['COUNTRYCODE']==c,'EPI_GENUINE_PEAK_2_DATE'].values[0]
            if np.isnan(peak_2_date):
                S_ = np.argmin(data.loc[(data['countrycode']==c)&(data['date']>peak_1_date),'new_per_day_smooth'])
            else:
                S_ = np.argmin(data.loc[(data['countrycode']==c)&(data['date']>peak_1_date)&(data['date']<peak_2_date),'new_per_day_smooth'])
            S_date = data.iloc[S_]['date']
            S_confirmed = data.iloc[S_]['confirmed']
            S0 = min(data.loc[(data['countrycode']==c)&(data['date']>S_date)&(data['confirmed']>=S_confirmed+50),'date'])
            data.loc[data['countrycode']==c,'S0'] = S0
            data.loc[data['countrycode']==c,'s'] = [a.days for a in (data.loc[data['countrycode']==c,'date'] - S0)]
        except:
            print('Error: ' +  c)
            data.loc[data['countrycode']==c,'s'] = np.nan
    else:
        data.loc[data['countrycode']==c,'S0'] = np.nan
        data.loc[data['countrycode']==c,'s'] = np.nan


#%% Scatterplot of duration of first wave vs. duration of stringency
data_plot=FINAL.loc[FINAL['CLASS']!=1,['GOV_PEAK_1_WIDTH','EPI_GENUINE_PEAK_1_WIDTH','CLASS_LABEL']].dropna(how='any')

plt.clf()
plt.figure(figsize = (15,10))
sns.set_style("darkgrid")
g = sns.scatterplot(x='GOV_PEAK_1_WIDTH',y='EPI_GENUINE_PEAK_1_WIDTH',data=data_plot,
                    hue='CLASS_LABEL',palette=sns.color_palette("muted", data_plot.CLASS_LABEL.nunique()),
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
        data_plot = epidemiology_results.loc[epidemiology_results['countrycode']==country,:]
        ax1.plot(data_plot['date'].values, data_plot['new_per_day'].values, label='New Cases per Day')
        ax1.plot(data_plot['date'].values, data_plot['new_per_day_smooth'].values, label='New Cases per Day (Smoothed)', 
                 color='black', linestyle='dashed')
        # GOV: Stringency Index
        ax2 = ax1.twinx()
        data_plot = government_response.loc[government_response['countrycode']==country,:]
        ax2.plot(data_plot['date'].values, data_plot['stringency_index'].values, label='Stringency Index', 
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

#%% Figure 2: Plot of average stringency index over time for each class
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
handles, labels = g.get_legend_handles_labels()
g.legend(handles=handles[1:], labels=['Entering First Wave','Past First Wave','Entering Second Wave'])
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
handles, labels = g.get_legend_handles_labels()
g.legend(handles=handles[1:], labels=['First Wave','Second Wave'])
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
    

#%% Figure 4: time series of EPI, GOV and MOB

for c in data['CLASS'].unique():
    if c == 0 or c == 4:
        continue

    plt.close('all')    
    plt.clf()
    f, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    f.tight_layout(rect=[0, 0, 1, 0.98], pad=3)
    sns.set_palette("husl")

    countries = FINAL[FINAL['CLASS'] == c]['COUNTRYCODE'].unique()

    temp = FINAL.loc[FINAL['CLASS']==c,['GOV_C6_RAISED_DATE','T0']].dropna()
    avg_raised = np.mean([(a-b).days for a, b in zip(temp['GOV_C6_RAISED_DATE'],temp['T0'])])
    n1 = len(FINAL.loc[FINAL['CLASS']==c,['GOV_C6_RAISED_DATE']].dropna())
    
    temp = FINAL.loc[FINAL['CLASS']==c,['GOV_C6_LOWERED_DATE','T0']].dropna()
    avg_lowered = np.mean([(a-b).days for a, b in zip(temp['GOV_C6_LOWERED_DATE'],temp['T0'])])
    n2 = len(FINAL.loc[FINAL['CLASS']==c,['GOV_C6_LOWERED_DATE']].dropna())
    
    temp = FINAL.loc[FINAL['CLASS']==c,['GOV_C6_RAISED_AGAIN_DATE','T0']].dropna()
    avg_raised_again = np.mean([(a-b).days for a, b in zip(temp['GOV_C6_RAISED_AGAIN_DATE'],temp['T0'])])
    n3 = len(FINAL.loc[FINAL['CLASS']==c,['GOV_C6_RAISED_AGAIN_DATE']].dropna())
    
    ten_countries = list(pd.Series({country:max(data[data['COUNTRYCODE']==country]['confirmed'].dropna()) \
                           for country in countries}).nlargest(n = 10).index)

    data_all = data.loc[data['COUNTRYCODE'].isin(countries),['COUNTRYCODE','t','new_per_day','new_per_day_smooth','new_per_day_ma', \
                                                             'new_per_day_per10k','new_per_day_smooth_per10k','new_per_day_ma_per10k', \
                                                             'residential','residential_smooth','stringency_index']]
    data_ten = data.loc[data['COUNTRYCODE'].isin(ten_countries),['COUNTRYCODE','t','new_per_day','new_per_day_smooth','new_per_day_ma', \
                                                                 'new_per_day_per10k','new_per_day_smooth_per10k','new_per_day_ma_per10k', \
                                                                 'residential','residential_smooth','stringency_index']]

    # restrict the date range: keep only t values with at least 80% of the countries present
    ns = {t:len(data_all.loc[data_all['t']==t,'new_per_day'].dropna()) for t in data_all['t'].unique()}
    ns = [t for t in ns.keys() if ns[t] >= 0.95*len(data_all['COUNTRYCODE'].unique())]
    t_lower_lim = min(ns)
    t_upper_lim = max(ns)-1
    data_all = data_all[(data_all['t']>=t_lower_lim) & (data_all['t']<=t_upper_lim)]
    data_ten = data_ten[(data_ten['t']>=t_lower_lim) & (data_ten['t']<=t_upper_lim)]
    
    # scale each country's new cases curve by the max value of the aggregate curve
    data_epi = data_all[['t','new_per_day_smooth_per10k']].dropna()
    data_epi = data_epi.groupby(by=['t']).mean()
    max_epi = max(data_epi['new_per_day_smooth_per10k'])
    for country in ten_countries:
        temp_max_epi = max(data_ten.loc[data_ten['COUNTRYCODE']==country,'new_per_day'])
        data_ten.loc[data_ten['COUNTRYCODE']==country,'new_per_day_scaled'] = data_ten.loc[data_ten['COUNTRYCODE']==country,'new_per_day']*(max_epi/temp_max_epi)
        temp_max_epi = max(data_ten.loc[data_ten['COUNTRYCODE']==country,'new_per_day_smooth'])
        data_ten.loc[data_ten['COUNTRYCODE']==country,'new_per_day_smooth_scaled'] = data_ten.loc[data_ten['COUNTRYCODE']==country,'new_per_day_smooth']*max_epi/temp_max_epi
    
    # plot aggregate curve for all countries in class
    #sns.lineplot(x = 't',y = 'new_per_day_per10k', data=data_all, color = 'black', ci = None, label = 'aggregate', ax=axes[0])
    #sns.lineplot(x = 't',y = 'new_per_day_ma_per10k', data=data_all, color = 'black', ci = None, label = 'aggregate', ax=axes[0])
    sns.lineplot(x = 't',y = 'new_per_day_smooth_per10k', data=data_all, color = 'black', ci = None, label = 'Aggregate', ax=axes[0], legend=False)
    sns.lineplot(x = 't',y = 'stringency_index', data=data_all, color = 'black', ci = None, label = 'Aggregate', ax=axes[1])
    sns.lineplot(x = 't',y = 'residential_smooth', data=data_all, color = 'black', ci = None, label = 'Aggregate', ax=axes[2], legend=False)
        
    # plot new cases per day smoothed curve for top 10 countries
    ax2 = axes[0].twinx()
    sns.lineplot(x='t',y='new_per_day_smooth_per10k',data=data_ten, hue='COUNTRYCODE', ax=ax2,alpha=0.6,legend=False)
    sns.lineplot(x='t',y='stringency_index',data=data_ten, hue='COUNTRYCODE', ax=axes[1],alpha=0.6)
    sns.lineplot(x='t',y='residential_smooth',data=data_ten, hue='COUNTRYCODE', ax=axes[2],alpha=0.6, legend=False)

    # vertical lines of stay at home flag raised/lowered/raised again
    maxes={0:axes[0].get_ylim()[1], 1:axes[1].get_ylim()[1], 2:axes[2].get_ylim()[1]}
    for i in [0,1,2]:
        axes[i].axvline(avg_raised, linestyle='dashed', color='red')
        axes[i].text(avg_raised+1, 0.95*maxes[i], 'Flag Raised')
        axes[i].text(avg_raised+1, 0.9*maxes[i], 'n = ' + str(n1))
    if c >= 2:
        for i in [0,1,2]:
            axes[i].axvline(avg_lowered, linestyle='dashed', color='red')
            axes[i].text(avg_lowered+1, 0.95*maxes[i], 'Flag Lowered')
            axes[i].text(avg_lowered+1, 0.9*maxes[i], 'n = ' + str(n2))
    if c >= 3:
        for i in [0,1,2]:
            axes[i].axvline(avg_raised_again, linestyle='dashed', color='red')
            axes[i].text(avg_raised_again+1, 0.95*maxes[i], 'Flag Raised Again')
            axes[i].text(avg_raised_again+1, 0.9*maxes[i], 'n = ' + str(n3))

    axes[0].set_xlim([t_lower_lim,t_upper_lim])
    axes[0].set_title('New Cases per Day Over Time')
    axes[0].set_xlabel('Days Since T0 (First Day of 50 Total Cases)')
    axes[0].set_ylabel('New Cases per Day per 10,000 (Aggregate)')
    ax2.set_ylabel('New Cases per Day per 10,000 (Individual Countries)')
    ax2.grid(None)
    axes[1].set_xlim([t_lower_lim,t_upper_lim])
    axes[1].set_title('Government Response Over Time')
    axes[1].set_xlabel('Days Since T0 (First Day of 50 Total Cases)')
    axes[1].set_ylabel('Stringency Index')
    axes[2].set_xlim([t_lower_lim,t_upper_lim])
    axes[2].set_title('Residential Mobility Over Time')
    axes[2].set_xlabel('Days Since T0 (First Day of 50 Total Cases)')
    axes[2].set_ylabel('Residential Mobility (% change from baseline)')
    class_names = {1:'Entering First Wave',2:'Past First Wave',3:'Entering Second Wave'}
    f.suptitle('Figure 4: New Cases, Government Response and Residential Mobility Over Time for Countries '+class_names[c])
    
    for i in [axes[0],axes[1],axes[2]]:
        box = i.get_position()
        i.set_position([box.x0, box.y0, box.width*0.86, box.height])
    
    axes[1].legend().set_visible(False)
    lines, labels = axes[1].get_legend_handles_labels()
    labels[1] = 'Countries'
    for i in range(2,12):
        labels[i] = FINAL.loc[FINAL['COUNTRYCODE']==labels[i],'COUNTRY'].values[0]
    f.legend(lines, labels, loc = (0.88,0.75))
    plt.gcf().text(0.13, 0.008, "Note: data for new cases per day and residential mobility have been smoothed using a spline fit approximation to reduce measurement noise.")

    if SAVE_PLOTS:
        plt.savefig(PATH + 'fig_4_stage_' + str(int(c)) + '_timeline.png')


#%% Testing: plot of epi, gov, mob comparing the first and second wave

plt.close('all')    
plt.clf()
f, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
f.tight_layout(rect=[0, 0, 1, 0.98], pad=3)
sns.set_palette("husl")

countries = FINAL[FINAL['CLASS'] >= 3]['COUNTRYCODE'].unique()

data_T = data.loc[(data['COUNTRYCODE'].isin(countries))&(data['date']<=data['S0']),
                  ['COUNTRYCODE','t','S0','new_per_day','new_per_day_smooth','new_per_day_ma', \
                   'new_per_day_per10k','new_per_day_smooth_per10k','new_per_day_ma_per10k', \
                   'residential','residential_smooth','stringency_index']]
data_S = data.loc[(data['COUNTRYCODE'].isin(countries))&(data['date']>=data['S0']),
                  ['COUNTRYCODE','s','S0','new_per_day','new_per_day_smooth','new_per_day_ma', \
                   'new_per_day_per10k','new_per_day_smooth_per10k','new_per_day_ma_per10k', \
                   'residential','residential_smooth','stringency_index']]
    
# restrict the date range: keep only t values with at least 80% of the countries present
ns = {t:len(data_all.loc[data_all['t']==t,'new_per_day'].dropna()) for t in data_all['t'].unique()}
ns = [t for t in ns.keys() if ns[t] >= 0.95*len(data_all['COUNTRYCODE'].unique())]
t_lower_lim = min(ns)
t_upper_lim = max(ns)-1
#data_all = data_all[(data_all['t']>=t_lower_lim) & (data_all['t']<=t_upper_lim)]

# plot aggregate curve for all countries in class
sns.lineplot(x = 't',y = 'new_per_day_smooth_per10k', data=data_T, color = 'red', ci = None, label = 'Aggregate - First Wave', ax=axes[0])
sns.lineplot(x = 't',y = 'stringency_index', data=data_T, color = 'red', ci = None, label = 'Aggregate - First Wave', ax=axes[1], legend=False)
sns.lineplot(x = 't',y = 'residential_smooth', data=data_T, color = 'red', ci = None, label = 'Aggregate - First Wave', ax=axes[2], legend=False)

sns.lineplot(x = 's',y = 'new_per_day_smooth_per10k', data=data_S, color = 'blue', ci = None, label = 'Aggregate - Second Wave', ax=axes[0])
sns.lineplot(x = 's',y = 'stringency_index', data=data_S, color = 'blue', ci = None, label = 'Aggregate - Second Wave', ax=axes[1], legend=False)
sns.lineplot(x = 's',y = 'residential_smooth', data=data_S, color = 'blue', ci = None, label = 'Aggregate - Second Wave', ax=axes[2], legend=False)

#axes[0].set_xlim([t_lower_lim,t_upper_lim])
axes[0].set_title('New Cases per Day Over Time')
axes[0].set_xlabel('Days Since T0 (start of first wave) or S0 (start of second wave)')
axes[0].set_ylabel('New Cases per Day per 10,000 (Aggregate)')
ax2.set_ylabel('New Cases per Day per 10,000 (Individual Countries)')
#axes[1].set_xlim([t_lower_lim,t_upper_lim])
axes[1].set_title('Government Response Over Time')
axes[1].set_xlabel('Days Since T0 (start of first wave) or S0 (start of second wave)')
axes[1].set_ylabel('Stringency Index')
#axes[2].set_xlim([t_lower_lim,t_upper_lim])
axes[2].set_title('Residential Mobility Over Time')
axes[2].set_xlabel('Days Since T0 (start of first wave) or S0 (start of second wave)')
axes[2].set_ylabel('Residential Mobility (% change from baseline)')
f.suptitle('New Cases, Government Response and Residential Mobility Over Time for Countries In Second Wave, Comparing First vs. Second Wave')


#%% Figure 3: Scatterplot of response time vs. number of cumulative cases
data=FINAL.loc[FINAL['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave','Entering_Second_Wave','Past_Second_Wave']) \
               ,['GOV_MAX_SI_DAYS_FROM_T0','GOV_MAX_SI','CLASS_LABEL','T0','POPULATION','EPI_CONFIRMED','CFR',]].dropna(how='any')

data.loc[data['CLASS_LABEL'].isin(['Entering_First_Wave','Past_First_Wave']),'CLASS_COARSE'] = 'First_Wave'
data.loc[data['CLASS_LABEL'].isin(['Entering_Second_Wave','Past_Second_Wave']),'CLASS_COARSE'] = 'Second_Wave'

data['EPI_CONFIRMED_PER_10K'] = 10000*data['EPI_CONFIRMED']/data['POPULATION'] 
data['EPI_LOG_CONFIRMED_PC'] = np.log(data['EPI_CONFIRMED']/data['POPULATION'])
data['EPI_LOG_CONFIRMED'] = np.log(data['EPI_CONFIRMED'])

plt.close('all')
plt.clf()
plt.figure(figsize = (10,7))
sns.set_style("darkgrid")
g = sns.scatterplot(x='GOV_MAX_SI_DAYS_FROM_T0',y='EPI_CONFIRMED_PER_10K',data=data,
                    hue='CLASS_COARSE',palette=sns.color_palette("muted", data.CLASS_COARSE.nunique()),
                    hue_order = ['First_Wave','Second_Wave'],
                    style='CLASS_COARSE', s=50)
plt.title('Government Response Time Against Number of Confirmed Cases')
plt.xlabel('Response Time: Days from T0 to Peak Date of Stringency Index')
plt.ylabel('Cumulative Number of Confirmed Cases per 10,000')
handles, labels = g.get_legend_handles_labels()
g.legend(handles=handles[1:], labels=labels[1:])
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

