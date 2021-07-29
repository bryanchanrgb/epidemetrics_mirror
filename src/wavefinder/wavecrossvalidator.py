"""
NAME
    WaveCrossValidator

DESCRIPTION
    A WaveCrossValidator object can compare waves from two different time series and use one series to impute the
    presence of waves in the second.

ATTRIBUTES
    title (str): A title for labelling plots.

METHODS
    __init__: Sets the variable title.
    apply: Identifies additional waves in a time series based on those identified in a reference time series.
    run: Extracts raw data, identified waves and relevant parameters from two WaveList objects to pass to apply.
"""

import numpy as np
from pandas import DataFrame, Series

from wavefinder.wavelist import WaveList
import wavefinder.utils.trough_finder as trough_finder
from wavefinder.waveplotter import plot_cross_validator


class WaveCrossValidator:
    def __init__(self, title: str):
        """ Sets the title for plotting purposes. """
        self.title = title

    @staticmethod
    def apply(raw_data: Series, input_sub_b: DataFrame, input_sub_c: DataFrame,
              reference_sub_c: DataFrame, prominence_threshold: float,
              proportional_prominence_threshold: float) -> DataFrame:
        """
        Where the waves in input_sub_c are not aligned with those in reference_sub_c, this method imputes additional
        waves in input_sub_c.

        Parameters:
            raw_data (Series): The time series where additional waves are to be imputed.
            input_sub_b (DataFrame): The waves found in raw_data after Sub-Algorithm B, from which additional waves
            will be drawn to match those in reference_sub_c.
            input_sub_c (DataFrame): The waves found in raw_data after Sub-Algorithm C.
            reference_sub_c (DataFrame): The waves found in the second time series after Sub-Algorithm C, which will
            be used to impute additional waves in raw_data.
            prominence_threshold (float): The minimum prominence which a wave must have. Used to determine if a final
            trough is present in the data.
            proportional_prominence_threshold (float): The minimum prominence which a peak must have, as a ratio of the
            value at the peak. Used to determine if a final trough is present in the data.

        Returns:
            apply(raw_data, input_sub_b, input_sub_c, reference_sub_c, prominence_threshold,
            proportional_prominence_threshold): A DataFrame with the peaks and troughs
            in raw_data after adding back waves from input_sub_c matching those in reference_sub_c, but not present
            in input_sub_c.
        """

        reference_peaks = reference_sub_c[reference_sub_c.peak_ind == 1].location
        reference_troughs = reference_sub_c[reference_sub_c.peak_ind == 0].location
        results = input_sub_c.copy()

        # iterate through the waves in the reference time series
        for i, reference_peak in enumerate(reference_peaks):
            # identify the start and end of the wave
            window_start = reference_troughs.iloc[i - 1] if i > 0 else 0
            window_end = reference_troughs.iloc[i] if i < len(reference_troughs) else raw_data.index[-1]
            # check if a peak in the first series already exists during the wave - if it does then continue
            if np.any([True if (x >= window_start) and (x <= window_end)
                       else False for x in input_sub_c.loc[input_sub_c['peak_ind'] == 1, 'location']]):
                continue
            # if there is peak in input_sub_b during the wave, use the highest one
            candidates = input_sub_b[(input_sub_b['peak_ind'] == 1) &
                                     (input_sub_b['location'] >= window_start) & (
                                             input_sub_b['location'] <= window_end)]
            if len(candidates) > 0:
                new_peak = candidates.loc[candidates.idxmax()['y_position']]
                results = results.append(new_peak)

        # next add back any troughs between peaks
        result_troughs = input_sub_b[input_sub_b.peak_ind == 0]
        results = results[results.peak_ind == 1]
        results = trough_finder.run(results, result_troughs, raw_data, prominence_threshold,
                                    proportional_prominence_threshold)

        return results

    def run(self, input_wavelist: WaveList, reference_wavelist: WaveList, plot: bool = False,
            plot_path: str = '') -> DataFrame:
        """
        Imputes the presence of additional waves in the input_wavelist from those in the reference_wavelist.

        Parameters:
            input_wavelist (WaveList): The WaveList object where additional waves are to be imputed
            reference_wavelist (WaveList): The WaveList object which will be used to impute the additional waves.
            plot (bool): Whether the output should be plotted.
            plot_path (str): Location to store plots.

        Returns:
            run(input_wavelist, reference_wavelist, plot, plot_path): A DataFrame with the peaks and troughs after
            cross-validation.
        """

        # extract relevant data and parameters from the WaveList objects
        input_data = input_wavelist.raw_data
        prominence_threshold = input_wavelist.prominence_threshold
        proportional_prominence_threshold = input_wavelist.prominence_height_threshold

        # pass these to the apply method
        results = self.apply(input_data, input_wavelist.peaks_sub_b, input_wavelist.peaks_sub_c,
                             reference_wavelist.peaks_sub_c, prominence_threshold, proportional_prominence_threshold)

        # plot the results if required
        if plot:
            plot_cross_validator(input_wavelist, reference_wavelist, results, self.title, plot_path)

        return results
