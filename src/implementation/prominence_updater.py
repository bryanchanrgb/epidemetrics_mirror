import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks


class ProminenceUpdater:
    def __init__(self, data, field):
        self.initial_value = data[field].iloc[0]
        self.terminal_value = data[field].iloc[-1]
        self.initial_location = min(data.index) - 1
        self.terminal_location = max(data.index) + 1
        self.endpoints = pd.DataFrame([[self.initial_location, np.nan, self.initial_value, np.nan],
                                       [self.terminal_location, np.nan, self.terminal_value, np.nan]],
                                      columns=['location', 'prominence', 'y_position', 'peak_ind'])

    def makeframe(self, data, peak, peak_properties, trough, trough_properties):
        peaks = data.loc[peak]
        peaks['prominence'] = peak_properties['prominences']
        peaks['peak_ind'].values[:] = 1

        troughs = data.loc[trough]
        troughs['prominence'] = trough_properties['prominences']
        troughs['peak_ind'].values[:] = 0

        results = pd.concat([peaks, troughs])

        results = results.sort_values(by='location').reset_index(drop=True)
        return results

    def run(self, data) -> DataFrame:
        ''' take a list of peaks and recalculate the prominence '''
        data = data[['location', 'prominence', 'y_position', 'peak_ind']]
        data = pd.concat([data, self.endpoints])
        data = data.sort_values(by='location').reset_index(drop=True)
        y_vals = data.y_position.tolist()
        peak, peak_properties = find_peaks(y_vals, prominence=0, distance=1)
        trough, trough_properties = find_peaks([-y for y in y_vals], prominence=0, distance=1)

        results = self.makeframe(data, peak, peak_properties, trough, trough_properties)
        return results
