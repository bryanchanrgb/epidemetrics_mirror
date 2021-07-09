import numpy as np
import pandas as pd
from pandas import DataFrame
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater
from data_provider import DataProvider


class AlgorithmSpikeCleaner:
    def __init__(self, config: Config, data_provider: DataProvider):
        self.spike_sensitivity = config.spike_sensitivity
        self.ma_window = data_provider.ma_window

    @staticmethod
    def apply(data: DataFrame, prominence_updater: ProminenceUpdater, spike_sensitivity: float,
              ma_window: int) -> DataFrame:
        # calculate the slopes of each min-max pair wrt log-scale
        min_non_zero = 1 / ma_window
        data['log_y'] = np.log(data['y_position'].mask(data['y_position'] < min_non_zero, min_non_zero))
        data['slopes'] = abs(data['log_y'].diff() / data['location'].diff())
        spike_cutoff = data['slopes'].mean() + spike_sensitivity * data['slopes'].std()
        while data['slopes'].max() > spike_cutoff:
            i = data.idxmax()['slopes']
            data.drop(index=[i - 1, i], inplace=True)
            data.reset_index(drop=True, inplace=True)
            data['slopes'] = abs(data['log_y'].diff() / data['location'].diff())
        data.drop(columns=['log_y', 'slopes'], inplace=True)
        data = prominence_updater.run(data)
        return data

    def run(self, input_data_df: DataFrame, prominence_updater: ProminenceUpdater = None) -> DataFrame:
        output_data_df = AlgorithmSpikeCleaner.apply(input_data_df, prominence_updater, self.spike_sensitivity,
                                                     self.ma_window)
        return output_data_df
