import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks


class ProminenceUpdater:
    """
    NAME
        ProminenceUpdater

    DESCRIPTION
        A ProminenceUpdater object implements a method for recalculating the prominence of peaks and troughs in a time series as different waves are merged.

    ATTRIBUTES
        endpoints (DataFrame): The locations and value of the first and last elements of the time series.

    METHODS
        __init__: Creates endpoints from a time series.
        makeframe: Combines a list of peaks and a list of troughs into a DataFrame formatted for use in a WaveList object.
        run: Takes a list of peaks and troughs, removes any redundant entries, and calculates the prominences.
    """

    def __init__(self, data):
        """ Extract first and last element from a Series for use in run. """
        initial_value = data.iloc[0]
        terminal_value = data.iloc[-1]
        initial_location = min(data.index) - 1
        terminal_location = max(data.index) + 1
        self.endpoints = pd.DataFrame([[initial_location, np.nan, initial_value, np.nan],
                                       [terminal_location, np.nan, terminal_value, np.nan]],
                                      columns=['location', 'prominence', 'y_position', 'peak_ind'])

    @staticmethod
    def makeframe(data, peak, peak_properties, trough, trough_properties):
        """
        Combines a list of peaks and troughs returned by scipy.signal.find_peaks into a suitably formatted DataFrame

        Parameters:
            data (DataFrame): The data from which peaks and troughs were recalculated
            peak (Lst): A list of indices of peaks calculated by scipy.signal.find_peaks
            peak_properties (Dict): The properties of those peaks as returned from scipy.signal.find_peaks
            trough (Lst): A list of indices of troughs calculated by scipy.signal.find_peaks
            trough_properties (Dict): The properties of those troughs as returned from scipy.signal.find_peaks

        Returns:
            makeframe(data, peak, peak_properties, trough, trough_properties): A DataFrame containing the peaks and troughs found in data, along with their locations, prominences and values.
        """

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
        """ take a list of peaks and recalculate the prominence """
        data = data[['location', 'prominence', 'y_position', 'peak_ind']]
        data = pd.concat([data, self.endpoints])
        data = data.sort_values(by='location').reset_index(drop=True)
        y_vals = data.y_position.tolist()
        peak, peak_properties = find_peaks(y_vals, prominence=0, distance=1)
        trough, trough_properties = find_peaks([-y for y in y_vals], prominence=0, distance=1)

        results = self.makeframe(data, peak, peak_properties, trough, trough_properties)
        return results
