import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from implementation.config import Config
from data_provider import DataProvider


class AlgorithmA:
    def __init__(self, config: Config, data_provider: DataProvider):
        self.config = config
        self.data_provider = data_provider

    def run(self, country, field='new_per_day_smooth', plot=False, override=None) -> DataFrame:
        if type(override) == pd.core.frame.DataFrame:
            data = override.copy()
        else:
            data = self.data_provider.get_series(country, field)
        peak = find_peaks(data[field].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in data[field].values], prominence=0, distance=1)
        sub_a = pd.DataFrame(data=np.transpose([np.append(data.index[peak[0]], data.index[trough[0]]),
                                                np.append(peak[1]['prominences'], trough[1]['prominences']),
                                                np.append(data.index[peak[1]['left_bases']],
                                                          data.index[trough[1]['left_bases']]),
                                                np.append(data.index[peak[1]['right_bases']],
                                                          data.index[trough[1]['right_bases']])]),
                             columns=['location', 'prominence', 'left_base', 'right_base'])
        sub_a['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
        sub_a = sub_a.sort_values(by='location').reset_index(drop=True)
        # if there are fewer than 3 points, the algorithm cannot be run, return nothing
        if len(sub_a) < 3:
            results = pd.DataFrame(
                columns=['location', 'prominence', 'duration', 'left_base', 'right_base', 'index', 'peak_ind'])
        else:
            # calculate the duration of extrema i as the distance between the extrema to the left and to the right of i
            for i in range(1, len(sub_a) - 1):
                sub_a.loc[i, 'duration'] = sub_a.loc[i + 1, 'location'] - sub_a.loc[i - 1, 'location']
            # remove peaks and troughs until the smallest duration meets T_SEP
            while np.nanmin(sub_a['duration']) < self.config.t_sep_a and len(sub_a) >= 3:
                # sort the peak/trough candidates by prominence, retaining the location index
                sub_a = sub_a.sort_values(by=['prominence', 'duration']).reset_index(drop=False)
                # remove the lowest prominence candidate with duration < T_SEP
                x = min(sub_a[sub_a['duration'] < self.config.t_sep_a].index)
                i = sub_a.loc[x, 'index']
                sub_a.drop(index=x, inplace=True)
                # remove whichever adjacent candidate has the lower prominence. If tied, remove the earlier.
                if sub_a.loc[sub_a['index'] == i + 1, 'prominence'].values[0] >= \
                        sub_a.loc[sub_a['index'] == i - 1, 'prominence'].values[0]:
                    sub_a = sub_a.loc[
                        sub_a['index'] != i - 1, ['prominence', 'location', 'peak_ind', 'left_base', 'right_base']]
                else:
                    sub_a = sub_a.loc[
                        sub_a['index'] != i + 1, ['prominence', 'location', 'peak_ind', 'left_base', 'right_base']]
                # re-sort by location and recalculate duration
                sub_a = sub_a.sort_values(by='location').reset_index(drop=True)
                if len(sub_a) >= 3:
                    for i in range(1, len(sub_a) - 1):
                        sub_a.loc[i, 'duration'] = sub_a.loc[i + 1, 'location'] - sub_a.loc[i - 1, 'location']
                else:
                    sub_a['duration'] = np.nan
            results = sub_a.copy()

        if plot:
            self.plot(data, sub_a, field)
        # results returns a set of peaks and troughs which are at least a minimum distance apart
        return results

    def plot(self, data, sub_a, field):
        plt.plot(data[field].values)
        plt.scatter(sub_a['location'].values,
                    data[field].values[sub_a['location'].values.astype(int)], color='red', marker='o')
