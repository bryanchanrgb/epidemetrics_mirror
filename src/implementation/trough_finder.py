import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.signal import find_peaks
from implementation.config import Config


class TroughFinder:
    def __init__(self):
        pass

    @staticmethod
    def run(peak_list, trough_list, data, field, prominence_threshold, prominence_height_threshold):
        # between each remaining peak, retain the trough with the lowest value
        peak_list = peak_list.sort_values(by='location').reset_index(drop=True)
        results = peak_list
        for i in peak_list.index:
            if i < max(peak_list.index):
                wave_start = int(peak_list.loc[i, 'location'])
                wave_end = int(peak_list.loc[i+1, 'location'])
                candidate_troughs = trough_list[(trough_list['location'] >= wave_start) &
                                                (trough_list['location'] <= wave_end)]
                if len(candidate_troughs) > 0:
                    candidate_troughs = candidate_troughs.loc[candidate_troughs.idxmin()['y_position']]
                    results = results.append(candidate_troughs, ignore_index=True)
                else:
                    trough_idx = data[field].iloc[wave_start:wave_end].idxmin()['y_position']
                    candidate_troughs = pd.DataFrame([[trough_idx],[data[field].iloc[trough_idx]]], columns=[location, y_position])
                    results = results.append(candidate_troughs, ignore_index=True)
        results = results.sort_values(by='location').reset_index(drop=True)

        # add final trough after final peak
        if len(peak_list) > 0:
            candidate_troughs = trough_list[trough_list.location >= peak_list.location.iloc[-1]]
            if len(candidate_troughs) > 0:
                candidate_troughs = candidate_troughs.loc[candidate_troughs.idxmin()['y_position']]
                final_maximum = max(data[(data.index > candidate_troughs.location)][field])
                if (candidate_troughs.y_position <= (1 - prominence_height_threshold) *
                    peak_list.y_position.iloc[-1]) and (
                        peak_list.y_position.iloc[-1] - candidate_troughs.y_position >= prominence_threshold):
                    if (candidate_troughs.y_position <= (
                            1 - prominence_height_threshold) * final_maximum) and (
                            final_maximum - candidate_troughs.y_position >= prominence_threshold):
                        results = results.append(candidate_troughs, ignore_index=True)

        return results