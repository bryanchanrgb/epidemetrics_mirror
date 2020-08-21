'''
This sript generates the Figure 2 plot from the csv file fig_2_data.
Figure 2 is a line plot of stringency index over time for each country group.
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
    os.makedirs(PATH+'figures/',exist_ok=True)

# Read csv for data to plot
if READ_CSV:
    fig_2_data = pd.read_csv(PATH + 'fig_2_data.csv', index_col = 0)

# Set plot parameters
xlim_thresold = 0.9     # Plot will crop the x axis to only show t values with >=threshold proportion of countries present.
T0_threshold = 1000     # Number of total cases used to define T0. Must be set to the same value as used in generate_table.
include_class = ['EPI_ENTERING_FIRST','EPI_PAST_FIRST','EPI_ENTERING_SECOND'] # Classes to include in the plot. Past second is currently excluded due to the low sample size.

# Class labels to use in title
class_labels = {'EPI_OTHER': 'Other',
                'EPI_ENTERING_FIRST': 'Entering First Wave',
                'EPI_PAST_FIRST': 'Past First Wave',
                'EPI_ENTERING_SECOND': 'Entering Second Wave',
                'EPI_PAST_SECOND': 'Past Second Wave'}

# Set plot dimensions and style
plt.close('all')
plt.clf()
plt.figure(figsize = (15,7))
sns.set_style("darkgrid")
sns.set_palette('muted')

# Line plot of SI over time
g = sns.lineplot(x = 't', y ='stringency_index', data=fig_2_data[fig_2_data['CLASS_LABEL'].isin(include_class)],
                 hue = 'CLASS_LABEL', hue_order=include_class)

# Restrict the date range: keep only t values with at least x% of the countries present
ns = {t:len(fig_2_data.loc[fig_2_data['t']==t,'stringency_index'].dropna())
      for t in fig_2_data['t'].unique()}
ns = [t for t in ns.keys() if ns[t] >= xlim_thresold*len(fig_2_data['COUNTRYCODE'].unique())]
t_lower_lim = min(ns)
t_upper_lim = max(ns)
g.set(xlim=(t_lower_lim, t_upper_lim))

# Titles and labels
plt.title('Government Stringency Over Time for Each Country Cluster')
plt.xlabel('Days Since T0 (First Day of '+str(T0_threshold)+' Cumulative Cases)')
plt.ylabel('Stringency Index')
handles, labels = g.get_legend_handles_labels()
labels = [class_labels[labels[i]] for i in range(1,len(labels))]
g.legend(handles=handles[1:], labels=labels)


if SAVE_PLOTS:
    g.figure.savefig(PATH + 'figures/' + 'fig_2.png')