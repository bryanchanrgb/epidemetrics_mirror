'''
This sript generates the Figure 4 plot from the csv file fig_4_data.
Figure 4 is a line plot of new cases, government response and mobility over time.
'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

PATH = ''
SAVE_PLOTS = True
READ_CSV = True

# Create directory to save plot
if SAVE_PLOTS:
    os.makedirs(PATH+'charts/',exist_ok=True)

# Read csv for data to plot
if READ_CSV:
    fig_4_data = pd.read_csv(PATH + 'fig_4_data.csv', index_col = 0, 
                             parse_dates=['date','T0','GOV_C6_RAISED_DATE','GOV_C6_LOWERED_DATE','GOV_C6_RAISED_AGAIN_DATE'])

# Set plot parameters 
xlim_thresold = 0.8     # Plot will crop the x axis to only show t values with >=threshold proportion of countries present.
c = 'EPI_SECOND_WAVE'   # Plot only for countries entering or past second wave. Can be set to 'EPI_FIRST_WAVE' to plot first wave countries.
n_countries = 10        # Number of countries to show as individual time series.
T0_threshold = 1000     # Number of total cases used to define T0. Must be set to the same value as used in generate_table.

# Clear figures
plt.close('all')    
plt.clf()
# Set figure dimensions and style
f, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
f.tight_layout(rect=[0, 0, 1, 0.98], pad=3)
sns.set_palette("husl")
sns.set_style('darkgrid')

# Define the list of countries in the class (i.e. countries with a second wave)
countries = fig_4_data[fig_4_data['CLASS_COARSE'] == c]['COUNTRYCODE'].unique()
# Set the n highest countries by number of total cases to show individually
indiv_countries = list(pd.Series({country:max(fig_4_data[fig_4_data['COUNTRYCODE']==country]['confirmed'].dropna()) \
                       for country in countries}).nlargest(n = n_countries).index)

# Compute the average date the stay at home restrictions were raised and lowered
temp = fig_4_data.loc[fig_4_data['CLASS_COARSE']==c,['COUNTRYCODE','GOV_C6_RAISED_DATE','T0']].dropna().drop_duplicates()
avg_raised = np.mean([(a-b).days for a, b in zip(temp['GOV_C6_RAISED_DATE'],temp['T0'])])
n1 = len(temp)
temp = fig_4_data.loc[fig_4_data['CLASS_COARSE']==c,['COUNTRYCODE','GOV_C6_LOWERED_DATE','T0']].dropna().drop_duplicates()
avg_lowered = np.mean([(a-b).days for a, b in zip(temp['GOV_C6_LOWERED_DATE'],temp['T0'])])
n2 = len(temp)
temp = fig_4_data.loc[fig_4_data['CLASS_COARSE']==c,['COUNTRYCODE','GOV_C6_RAISED_AGAIN_DATE','T0']].dropna().drop_duplicates()
avg_raised_again = np.mean([(a-b).days for a, b in zip(temp['GOV_C6_RAISED_AGAIN_DATE'],temp['T0'])])
n3 = len(temp)

# Restrict the date range: keep only t values with at least x% of the countries present
ns = {t:len(fig_4_data.loc[(fig_4_data['COUNTRYCODE'].isin(countries))&(fig_4_data['t']==t),
                           ['new_per_day_smooth_per10k','stringency_index','residential_smooth']].dropna())
      for t in fig_4_data['t'].unique()}
ns = [t for t in ns.keys() if ns[t] >= xlim_thresold*len(countries)]
t_lower_lim = min(ns)
t_upper_lim = max(ns)

# Subset of data to plot
fig_4_data_agg = fig_4_data.loc[(fig_4_data['COUNTRYCODE'].isin(countries))&(fig_4_data['t']>=t_lower_lim)&(fig_4_data['t']<=t_upper_lim),
                                ['t','new_per_day_smooth_per10k','stringency_index','residential_smooth']]
fig_4_data_indiv = fig_4_data.loc[(fig_4_data['COUNTRYCODE'].isin(indiv_countries))&(fig_4_data['t']>=t_lower_lim)&(fig_4_data['t']<=t_upper_lim),
                                ['t','COUNTRYCODE','new_per_day_smooth_per10k','stringency_index','residential_smooth']]

# Plot aggregate smoothed curve for all countries in class
sns.lineplot(x = 't',y = 'new_per_day_smooth_per10k', data=fig_4_data_agg,
             color = 'black', ci = None, label = 'Aggregate', ax=axes[0], legend=False)
sns.lineplot(x = 't',y = 'stringency_index', data=fig_4_data_agg,
             color = 'black', ci = None, label = 'Aggregate', ax=axes[1])
sns.lineplot(x = 't',y = 'residential_smooth', data=fig_4_data_agg,
             color = 'black', ci = None, label = 'Aggregate', ax=axes[2], legend=False)
    
# Plot new cases per day smoothed curve for top 10 countries
ax2 = axes[0].twinx()
sns.lineplot(x='t',y='new_per_day_smooth_per10k',data=fig_4_data_indiv,
             hue='COUNTRYCODE', ax=ax2,alpha=0.6,legend=False)
sns.lineplot(x='t',y='stringency_index',data=fig_4_data_indiv,
             hue='COUNTRYCODE', ax=axes[1],alpha=0.6)
sns.lineplot(x='t',y='residential_smooth',data=fig_4_data_indiv,
             hue='COUNTRYCODE', ax=axes[2],alpha=0.6, legend=False)

# Plot vertical lines of stay at home flag raised/lowered/raised again
maxes={0:axes[0].get_ylim()[1], 1:axes[1].get_ylim()[1], 2:axes[2].get_ylim()[1]}
for i in [0,1,2]:
    axes[i].axvline(avg_raised, linestyle='dashed', color='black')
    axes[i].text(avg_raised+1, 0.95*maxes[i], 'Flag Raised')
    axes[i].text(avg_raised+1, 0.9*maxes[i], 'n = ' + str(n1))

    axes[i].axvline(avg_lowered, linestyle='dashed', color='black')
    axes[i].text(avg_lowered+1, 0.95*maxes[i], 'Flag Lowered')
    axes[i].text(avg_lowered+1, 0.9*maxes[i], 'n = ' + str(n2))

    axes[i].axvline(avg_raised_again, linestyle='dashed', color='black')
    axes[i].text(avg_raised_again+1, 0.95*maxes[i], 'Flag Raised Again')
    axes[i].text(avg_raised_again+1, 0.9*maxes[i], 'n = ' + str(n3))

# Set titles and limits
axes[0].set_xlim([t_lower_lim,t_upper_lim])
axes[0].set_title('New Cases per Day Over Time')
axes[0].set_xlabel('Days Since T0 (First Day of '+str(T0_threshold)+' Total Cases)')
axes[0].set_ylabel('New Cases per Day per 10,000 (Aggregate)')
ax2.set_ylabel('New Cases per Day per 10,000 (Individual Countries)')
ax2.grid(None)
axes[1].set_xlim([t_lower_lim,t_upper_lim])
axes[1].set_title('Government Response Over Time')
axes[1].set_xlabel('Days Since T0 (First Day of '+str(T0_threshold)+' Total Cases)')
axes[1].set_ylabel('Stringency Index')
axes[2].set_xlim([t_lower_lim,t_upper_lim])
axes[2].set_title('Residential Mobility Over Time')
axes[2].set_xlabel('Days Since T0 (First Day of '+str(T0_threshold)+' Total Cases)')
axes[2].set_ylabel('Residential Mobility (% change from baseline)')
f.suptitle('Figure 4: New Cases per Day, Government Response and Residential Mobility Over Time for Countries with a Second Wave')

# Set legend labels and position
for i in [axes[0],axes[1],axes[2]]:
    box = i.get_position()
    i.set_position([box.x0, box.y0, box.width*0.86, box.height])
axes[1].legend().set_visible(False)
lines, labels = axes[1].get_legend_handles_labels()
labels[1] = 'Countries'
for i in range(2,n_countries+2):
    labels[i] = fig_4_data.loc[fig_4_data['COUNTRYCODE']==labels[i],'COUNTRY'].values[0]
f.legend(lines, labels, loc = (0.88,0.70))
plt.gcf().text(0.13, 0.008, "Note: data for new cases per day and residential mobility have been smoothed using a spline fit approximation to reduce measurement noise.")

# Save Figure
if SAVE_PLOTS:
    plt.savefig(PATH + 'charts/' + 'fig_4.png')
