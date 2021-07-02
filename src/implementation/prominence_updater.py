import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks


class ProminenceUpdater:
    def __init__(self, initial_value, terminal_value):
        self.initial_value = initial_value
        self.terminal_value = terminal_value

    def run(self, data) -> DataFrame:
        ''' take a list of peaks and recalculate the prominence '''
        initial_location = min(data.location) - 1
        terminal_location = max(data.location) + 1
        endpoints = pd.DataFrame([[initial_location, self.initial_value], [terminal_location, self.terminal_value]],
                                 columns=['location', 'y_position'])
        data = pd.concat([data, endpoints])
        data = data.sort_values(by='location').reset_index(drop=True)
        peak, peak_properties = find_peaks(data.y_position.values, prominence=0, distance=1)
        trough, trough_properties = find_peaks([-x for x in data.y_position.values], prominence=0, distance=1)

        results = pd.DataFrame(data=np.transpose([np.append(data.location[peak], data.location[trough]),
                                                  np.append(peak_properties['prominences'],
                                                            trough_properties['prominences']),
                                                  np.append(data.location[peak_properties['left_bases']],
                                                            data.location[trough_properties['left_bases']]),
                                                  np.append(data.location[peak_properties['right_bases']],
                                                            data.location[trough_properties['right_bases']]),
                                                  np.append(data.y_position[peak], data.y_position[trough])]),
                               columns=['location', 'prominence', 'left_base', 'right_base', 'y_position'])
        results['peak_ind'] = np.append([1] * len(peak), [0] * len(trough))
        results = results.sort_values(by='location').reset_index(drop=True)
        return results
