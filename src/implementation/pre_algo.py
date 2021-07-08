import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater
from data_provider import DataProvider


class PreAlgo:
    def __init__(self, config: Config, data_provider: DataProvider):
        self.config = config
        self.data_provider = data_provider
        self.data = None

    def init_country(self, country, field='new_per_day_smooth') -> (DataFrame, ProminenceUpdater):
        self.data = self.data_provider.get_series(country, field)

        # initialise prominence_updater to run when pairs are removed
        prominence_updater = ProminenceUpdater(self.data, field)

        # identify initial list of peaks via find_peaks
        peak = find_peaks(self.data[field].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in self.data[field].values], prominence=0, distance=1)

        # collect into a single dataframe
        df = pd.DataFrame(data=np.transpose([np.append(self.data.index[peak[0]], self.data.index[trough[0]]),
                                             np.append(peak[1]['prominences'], trough[1]['prominences'])]),
                          columns=['location', 'prominence'])
        df['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
        df.loc[:, 'y_position'] = self.data[field][df['location']].values
        df = df.sort_values(by='location').reset_index(drop=True)
        return df, prominence_updater

    def clean_spikes(self, pre_algo, prominence_updater):
        # calculate the slopes of each min-max pair wrt log-scale
        data = pre_algo.copy()
        data['slopes'] = abs(np.log(data['y_position'].replace(0, 1)).diff() / data['location'].diff())
        spike_cutoff = data['slopes'].mean() + self.config.spike_sensitivity * data['slopes'].std()
        while data['slopes'].max() > spike_cutoff:
            i = data.idxmax()['slopes']
            data.drop(index = [i-1, i], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['slopes'] = abs(np.log(data['y_position'].replace(0, 1)).diff() / data['location'].diff())
        data.drop(columns = 'slopes', inplace=True)
        data = prominence_updater.run(data)
        return data

    def run(self, country: str, field: str = 'new_per_day_smooth', plot: bool = False) -> (DataFrame, DataFrame, ProminenceUpdater):
        pre_algo, prominence_updater = self.init_country(country, field=field)
        if plot:
            self.plot(pre_algo, field)
        spikes_removed = self.clean_spikes(pre_algo, prominence_updater)
        if plot:
            self.plot(spikes_removed, field)
        return pre_algo, spikes_removed, prominence_updater

    def plot(self, results: DataFrame, field: str):
        plt.plot(self.data[field].values)
        plt.scatter(results['location'].values, self.data[field].values[results['location'].values.astype(int)],
                    color='red', marker='o')
        plt.show()
