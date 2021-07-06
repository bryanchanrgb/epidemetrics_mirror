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

    def init_country(self, country, field='new_per_day_smooth') -> DataFrame:
        data = self.data_provider.get_series(country, field)

        # initialise prominence_updater to run when pairs are removed
        prominence_updater = ProminenceUpdater(data, field)

        # identify initial list of peaks via find_peaks
        peak = find_peaks(data[field].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in data[field].values], prominence=0, distance=1)

        # collect into a single dataframe
        df = pd.DataFrame(data=np.transpose([np.append(data.index[peak[0]], data.index[trough[0]]),
                                             np.append(peak[1]['prominences'], trough[1]['prominences'])]),
                          columns=['location', 'prominence'])
        df['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
        df.loc[:, 'y_position'] = data[field][df['location']].values
        df = df.sort_values(by='location').reset_index(drop=True)
        return df, prominence_updater

    @staticmethod
    def delete_pairs(data, t_sep_a):
        if np.nanmin(data['duration']) < t_sep_a and len(data) >= 3:
            # extract waves of low duration
            df1 = data[data['duration'] < t_sep_a]
            # extract those of least prominence
            df2 = df1[df1['prominence'] == min(df1['prominence'])]
            # find the shortest
            i = df2.idxmin()['duration']
            is_peak = data.loc[i, 'peak_ind'] - 0.5
            data.drop(index=i, inplace=True)
            # remove whichever adjacent candidate is a greater minimum, or a lesser maximum. If tied, remove the
            # earlier.
            if is_peak * (data.loc[i + 1, 'y_position'] - data.loc[i - 1, 'y_position']) >= 0:
                data.drop(index=i+1, inplace=True)
            else:
                data.drop(index=i-1, inplace=True)
        return data

    @staticmethod
    def apply(data: DataFrame, prominence_updater: ProminenceUpdater, t_sep_a: int) -> DataFrame:
        # if there are fewer than 3 points, the algorithm cannot be run, return list unchanged
        if len(data) < 3:
            return data

        else:
            # calculate the duration of extremum i as the distance between the extrema to the left and to the right of i
            data['duration'] = data['location'].diff(periods=1) - data['location'].diff(periods=-1)

        while np.nanmin(data['duration']) < t_sep_a and len(data) >= 3:
            # remove peaks and troughs until the smallest duration meets T_SEP
            data = AlgorithmA.delete_pairs(data, t_sep_a)
            # update prominence
            data = prominence_updater.run(data)
            # recalculate duration
            if len(data) >= 3:
                data['duration'] = data['location'].diff(periods=1) - data['location'].diff(periods=-1)
            else:
                break

        # results returns a set of peaks and troughs which are at least a minimum distance apart
        return data

    def run(self, country: str, field: str = 'new_per_day_smooth', plot: bool = False) -> DataFrame:
        data, prominence_updater = self.init_country(country, field=field)
        results = self.apply(data, prominence_updater, self.config.t_sep_a)
        if plot:
            self.plot(data, results, field)
        return results

    def plot(self, data: DataFrame, sub_a: DataFrame, field: str):
        plt.plot(data[field].values)
        plt.scatter(sub_a['location'].values, data[field].values[sub_a['location'].values.astype(int)],
                    color='red', marker='o')
        plt.show()
