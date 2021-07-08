import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater
from data_provider import DataProvider


class AlgorithmA:
    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def delete_pairs(data: DataFrame, t_sep_a: int) -> DataFrame:
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
                data.drop(index=i + 1, inplace=True)
            else:
                data.drop(index=i - 1, inplace=True)
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

    def run(self, input_data_df: DataFrame, prominence_updater: ProminenceUpdater = None) -> DataFrame:
        output_data_df = AlgorithmA.apply(input_data_df, prominence_updater, self.config.t_sep_a)
        return output_data_df
