
#%% -------------------------------------------------------------------------------------------------------------------- #
'''
FIGURE 1
'''
countries=['ZMB','GBR','GHA','CRI']
for country in countries:
    plt.close('all')
    country_series = figure_1_series[figure_1_series['countrycode'] == country].reset_index(drop=True)
    data = dict(figure_1_panel.loc[figure_1_panel['countrycode'] == country,].reset_index(drop=True).loc[0,])
    
    f, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    axes[0].plot(country_series['date'].values,
             country_series['new_per_day'].values,
             label='New Cases per Day')
    axes[0].plot(country_series['date'].values,
             country_series['new_per_day_smooth'].values,
             label='New Cases per Day 7 Day Moving Average')
    axes[1].plot(country_series['date'].values,
             country_series['dead_per_day'].values,
             label='Deaths per Day')
    axes[1].plot(country_series['date'].values,
             country_series['dead_per_day_smooth'].values,
             label='Deaths per Day 7 Day Moving Average')
    
    axes[0].plot(data['date_peak_1'], data['peak_1'], ".", ms=10, color='black')
    axes[0].plot(data['date_peak_2'], data['peak_2'], ".", ms=10, color='black')
    #axes[0].axvline(x=data['first_wave_start'])
    #axes[0].axvline(x=data['first_wave_end'])
    axes[0].axvspan(xmin=data['first_wave_start'], xmax=data['first_wave_end'],alpha=0.1,color='g')
    axes[0].axvspan(xmin=data['second_wave_start'], xmax=data['second_wave_end'],alpha=0.1,color='r')
    axes[1].axvspan(xmin=data['first_wave_start'], xmax=data['first_wave_end'],alpha=0.1,color='g')
    axes[1].axvspan(xmin=data['second_wave_start'], xmax=data['second_wave_end'],alpha=0.1,color='r')
    
    axes[0].text(data['date_peak_1'],0.9*axes[0].get_ylim()[1],'First Wave',ha='center')
    axes[0].text(data['date_peak_2'],0.9*axes[0].get_ylim()[1],'Second Wave',ha='center')
    axes[0].text(data['date_peak_1'],data['peak_1']+0.05*axes[0].get_ylim()[1],'First Peak: '+str(data['date_peak_1']))
    axes[0].text(data['date_peak_2'],data['peak_2']+0.05*axes[0].get_ylim()[1],'Second Peak: '+str(data['date_peak_2']))
    
    axes[0].set_title('New Cases per Day')
    axes[0].set_ylabel('New Cases per Day')
    axes[1].set_title('Deaths per Day')
    axes[1].set_ylabel('Deaths per Day')
    f.suptitle('Waves for ' + data['country'])
    plt.savefig(PLOT_PATH + 'figure_1_' + data['country'] + '.png')


