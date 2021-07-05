import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater
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

        # initialise prominence_updater to run when pairs are removed
        initial_value = data[field].iloc[0]
        terminal_value = data[field].iloc[-1]
        prominence_updater = ProminenceUpdater(initial_value, terminal_value)

        # identify initial list of peaks via find_peaks
        peak = find_peaks(data[field].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in data[field].values], prominence=0, distance=1)

        # collect into a single dataframe
        sub_a = pd.DataFrame(data=np.transpose([np.append(data.index[peak[0]], data.index[trough[0]]),
                                                np.append(peak[1]['prominences'], trough[1]['prominences']),
                                                np.append(data.index[peak[1]['left_bases']],
                                                          data.index[trough[1]['left_bases']]),
                                                np.append(data.index[peak[1]['right_bases']],
                                                          data.index[trough[1]['right_bases']])]),
                             columns=['location', 'prominence', 'left_base', 'right_base'])
        sub_a['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
        sub_a.loc[:, 'y_position'] = data[field][sub_a['location']].values
        sub_a = sub_a.sort_values(by='location').reset_index(drop=True)

        # if there are fewer than 3 points, the algorithm cannot be run, return list unchanged
        if len(sub_a) < 3:
            results = sub_a

        else:
            # calculate the duration of extremum i as the distance between the extrema to the left and to the right of i
            for i in range(1, len(sub_a) - 1):
                sub_a.loc[i, 'duration'] = sub_a.loc[i + 1, 'location'] - sub_a.loc[i - 1, 'location']
            # remove peaks and troughs until the smallest duration meets T_SEP
            while np.nanmin(sub_a['duration']) < self.config.t_sep_a and len(sub_a) >= 3:
                # sort the peak/trough candidates by prominence, retaining the location index
                sub_a = sub_a.sort_values(by=['prominence', 'duration']).reset_index(drop=False)
                # remove the lowest prominence candidate with duration < T_SEP
                x = min(sub_a[sub_a['duration'] < self.config.t_sep_a].index)
                i = sub_a.loc[x, 'index']
                is_peak = sub_a.loc[x, 'peak_ind'] - 0.5
                sub_a.drop(index=x, inplace=True)
                # remove whichever adjacent candidate is a greater minimum, or a lesser maximum. If tied, remove the
                # earlier.
                if is_peak * (sub_a.loc[sub_a['index'] == i + 1, 'y_position'].values[0] -
                              sub_a.loc[sub_a['index'] == i - 1, 'y_position'].values[0]) >= 0:
                    sub_a = sub_a.loc[
                        sub_a['index'] != i + 1, ['prominence', 'location', 'peak_ind', 'left_base', 'right_base',
                                                  'y_position']]
                else:
                    sub_a = sub_a.loc[
                        sub_a['index'] != i - 1, ['prominence', 'location', 'peak_ind', 'left_base', 'right_base',
                                                  'y_position']]
                # update prominence and recalculate duration
                sub_a = prominence_updater.run(sub_a)
                if len(sub_a) >= 3:
                    for i in range(1, len(sub_a) - 1):
                        sub_a.loc[i, 'duration'] = sub_a.loc[i + 1, 'location'] - sub_a.loc[i - 1, 'location']
                else:
                    break
            results = sub_a.copy()

        if plot:
            self.plot(data, sub_a, field)

        # results returns a set of peaks and troughs which are at least a minimum distance apart
        return results

    def plot(self, data, sub_a, field):
        plt.plot(data[field].values)
        plt.scatter(sub_a['location'].values,
                    data[field].values[sub_a['location'].values.astype(int)], color='red', marker='o')
        plt.show()
