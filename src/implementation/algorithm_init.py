import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater
from data_provider import DataProvider


class AlgorithmInit:
    def __init__(self, config: Config = None, data_provider: DataProvider = None):
        self.config = config
        self.data_provider = data_provider

    @staticmethod
    def init_country(data: DataFrame) -> (DataFrame, DataFrame):
        # identify initial list of peaks via find_peaks
        peak = find_peaks(data.values, prominence=0, distance=1)
        trough = find_peaks([-x for x in data.values], prominence=0, distance=1)

        # collect into a single dataframe
        df = pd.DataFrame(data=np.transpose([np.append(data.index[peak[0]], data.index[trough[0]]),
                                             np.append(peak[1]['prominences'], trough[1]['prominences'])]),
                          columns=['location', 'prominence'])
        df['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
        df.loc[:, 'y_position'] = data[df['location']].values
        df = df.sort_values(by='location').reset_index(drop=True)
        return df

    def run(self, country: str, field: str = 'new_per_day_smooth', plot: bool = False) -> (
            DataFrame, DataFrame, ProminenceUpdater):
        data = self.data_provider.get_series(country, field)
        if len(data) == 0:
            return data, None, None
        pre_algo = AlgorithmInit.init_country(data[field])
        prominence_updater = ProminenceUpdater(data, field)
        return data, pre_algo, prominence_updater
