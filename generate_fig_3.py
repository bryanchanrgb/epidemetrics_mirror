'''
This sript generates the Figure 3 plot from the csv file fig_3_data.
Figure 3 is a scatterplot of government response time against total cases.
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
    fig_3_data = pd.read_csv(PATH + 'fig_3_data.csv', index_col = 0)


plt.close('all')
plt.clf()
plt.figure(figsize = (10,7))
sns.set_style('darkgrid')
sns.set_palette('muted')

g = sns.scatterplot(x='GOV_MAX_SI_DAYS_FROM_T0',y='EPI_CONFIRMED_PER_10K',data=fig_3_data,
                    hue='CLASS_COARSE', style='CLASS_COARSE', s=50)
plt.title('Government Response Time Against Number of Confirmed Cases')
plt.xlabel('Response Time: Days from T0 to Peak Date of Stringency Index')
plt.ylabel('Total Number of Confirmed Cases per 10,000')
handles, labels = g.get_legend_handles_labels()
labels[1]='First Wave Countries'
labels[2]='Second Wave Countries'
g.legend(handles=handles[1:], labels=labels[1:])

if SAVE_PLOTS:
    g.figure.savefig(PATH + 'charts/' + 'fig_3.png')