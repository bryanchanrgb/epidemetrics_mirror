import numpy as np
from pandas import DataFrame, Series

from wavefinder.wavelist import WaveList
import wavefinder.utils.trough_finder as trough_finder
from wavefinder.waveplotter import plot_cross_validator


class WaveCrossValidator:
    def __init__(self, country: str):
        self.country = country

    @staticmethod
    def apply(raw_data: Series, input_sub_b: DataFrame, input_sub_c: DataFrame,
              reference_sub_c: DataFrame, prominence_threshold: float, prominence_height_threshold: float) -> DataFrame:

        # when a wave of deaths ends, check if there is match between the number of peaks in both series.
        # if not, then add the most prominent peak for the output of sub-algorithm B in the self.d_match days before
        # or after

        reference_peaks = reference_sub_c[reference_sub_c.peak_ind == 1].location
        reference_troughs = reference_sub_c[reference_sub_c.peak_ind == 0].location
        results = input_sub_c.copy()

        # iterate through the waves of deaths
        for i, reference_peak in enumerate(reference_peaks):
            # identify the start and end of the wave
            window_start = reference_troughs.iloc[i - 1] if i > 0 else 0
            window_end = reference_troughs.iloc[i] if i < len(reference_troughs) else raw_data.index[-1]
            # check if a case peak exists during the wave - if it does then continue
            if np.any([True if (x >= window_start) and (x <= window_end)
                       else False for x in input_sub_c.loc[input_sub_c['peak_ind'] == 1, 'location']]):
                continue
            # if peak in cases_sub_b output use the highest one
            candidates = input_sub_b[(input_sub_b['peak_ind'] == 1) &
                                     (input_sub_b['location'] >= window_start) & (
                                             input_sub_b['location'] <= window_end)]
            if len(candidates) > 0:
                new_peak = candidates.loc[candidates.idxmax()['y_position']]
                results = results.append(new_peak)

        # next add back any troughs as in algorithm_c
        result_troughs = input_sub_b[input_sub_b.peak_ind == 0]
        results = results[results.peak_ind == 1]
        results = trough_finder.run(results, result_troughs, raw_data, prominence_threshold,
                                    prominence_height_threshold)

        return results

    def run(self, input_wavelist: WaveList, reference_wavelist: WaveList, plot: bool = False,
            plot_path: str = '') -> DataFrame:
        input_data = input_wavelist.raw_data
        prominence_threshold = input_wavelist.prominence_threshold
        prominence_height_threshold = input_wavelist.prominence_height_threshold

        results = self.apply(input_data, input_wavelist.peaks_sub_b, input_wavelist.peaks_sub_c,
                             reference_wavelist.peaks_sub_c, prominence_threshold, prominence_height_threshold)
        if plot:
            plot_cross_validator(input_wavelist, reference_wavelist, self.country, results, plot_path)
        return results
