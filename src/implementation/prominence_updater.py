import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks


class ProminenceUpdater:
    def __init__(self, initial_value, terminal_value):
        self.initial_value = initial_value
        self.terminal_value = terminal_value

    def makeframe(self,data, peak, peak_properties, trough, trough_properties):
        results = pd.DataFrame(np.array([np.append(data.location[peak], data.location[trough]),
                                         np.append(peak_properties['prominences'],
                                                   trough_properties['prominences']),
                                         np.append(data.y_position[peak], data.y_position[trough]),
                                         np.append([1] * len(peak), [0] * len(trough))]).transpose(),
                               columns=['location', 'prominence', 'y_position', 'peak_ind'])


        results = results.sort_values(by='location').reset_index(drop=True)
        return results

    def run(self, data) -> DataFrame:
        ''' take a list of peaks and recalculate the prominence '''
        initial_location = min(data.location) - 1
        terminal_location = max(data.location) + 1
        endpoints = pd.DataFrame([[initial_location, np.nan, self.initial_value, np.nan], [terminal_location, np.nan, self.terminal_value, np.nan]],
                                 columns=['location', 'prominence', 'y_position', 'peak_ind'])
        data = pd.concat([data, endpoints])
        data = data.sort_values(by='location').reset_index(drop=True)
        y_vals = data.y_position.tolist()
        peak, peak_properties = find_peaks(y_vals, prominence=0, distance=1)
        trough, trough_properties = find_peaks([-y for y in y_vals], prominence=0, distance=1)

        results = self.makeframe(data, peak, peak_properties, trough, trough_properties)
        return results
