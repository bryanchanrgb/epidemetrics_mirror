import numpy as np
import pandas as pd
from pandas import DataFrame
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater
from data_provider import DataProvider


class AlgorithmSpikeCleaner:
    def __init__(self, data_provider: DataProvider, spike_cutoff: float, spike_width: float):
        self.spike_cutoff = spike_cutoff
        self.ma_window = data_provider.ma_window
        self.spike_width = spike_width * self.ma_window

    @staticmethod
    def apply(data: DataFrame, prominence_updater: ProminenceUpdater, spike_cutoff: float, spike_width: float,
              ma_window: int) -> DataFrame:
        # seems to need a new copy to avoid modifying in place
        data = data.copy()
        # calculate the slopes of each min-max pair wrt log-scale
        min_non_zero = 1 / ma_window
        data['log_y'] = np.log(data['y_position'].mask(data['y_position'] < min_non_zero, min_non_zero))
        data['slopes'] = data['log_y'].diff() / data['location'].diff()
        # keep a list of locations that were not spikes
        excluded_locations = []

        while len(data[~data['location'].isin(excluded_locations) & data['slopes'] > spike_cutoff]) > 1:
            # find largest positive slope
            i = data[~data['location'].isin(excluded_locations)].idxmax()['slopes']
            first_slope = data['slopes'].iloc[i]
            # find a large negative slope nearby
            nearby_data = data[abs(data['location'] - data['location'].iloc[i]) < spike_width * ma_window]
            j = nearby_data.idxmin()['slopes']
            # proceed to drop points if this slope is also high enough
            if (-1 * data['slopes'].iloc[j]) > spike_cutoff:
                # drop the points in between
                if j < i:
                    data.drop(index=list(range(j, i)), inplace=True)
                else:
                    data.drop(index=list(range(i, j)), inplace=True)
            else:
                # add the first_slope to a list to exclude and try again
                excluded_locations.append(data['location'].iloc[i])
            # recalculate the prominences and slopes
            data = prominence_updater.run(data)
            data['log_y'] = np.log(data['y_position'].mask(data['y_position'] < min_non_zero, min_non_zero))
            data['slopes'] = abs(data['log_y'].diff() / data['location'].diff())
        data.drop(columns=['log_y', 'slopes'], inplace=True)

        return data

    def run(self, input_data_df: DataFrame, prominence_updater: ProminenceUpdater = None) -> DataFrame:
        output_data_df = AlgorithmSpikeCleaner.apply(input_data_df, prominence_updater, self.spike_cutoff,
                                                     self.spike_width, self.ma_window)
        return output_data_df
