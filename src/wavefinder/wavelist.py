from __future__ import annotations
from pandas import DataFrame, Series
import numpy as np

import wavefinder.subalgorithms.algorithm_init as algorithm_init
import wavefinder.subalgorithms.algorithm_a as algorithm_a
import wavefinder.subalgorithms.algorithm_b as algorithm_b
import wavefinder.subalgorithms.algorithm_c_and_d as algorithm_c_and_d
import wavefinder.utils.trough_finder as trough_finder


class WaveList:
    """
    NAME
        WaveList

    DESCRIPTION
        A WaveList object holds a time series, the parameters used to find waves in the time series, and the waves
        identified at different stages of the algorithm.

    ATTRIBUTES
        raw_data (Series): The original data from which the peaks and troughs are identified.
        series_name (str): The name of the series, for labelling plots.
        t_sep_a (int): Threshold specifying minimum wave duration.
        prominence_threshold (float): The minimum prominence which a wave must have.
        proportional_prominence_threshold (float): The minimum prominence which a peak must have, as a ratio of the
        value at the peak.
        peaks_initial (DataFrame): The list of peaks and troughs in raw_data.
        peaks_sub_a (DataFrame): The list of peaks and troughs after Sub-Algorithm A has merged short waves.
        peaks_sub_b (DataFrame): The list of peaks and troughs after Sub-Algorithm B has merged short transient features.
        peaks_sub_c (DataFrame): The list of peaks and troughs after Sub-Algorithms C and D have merged less prominent waves.
        peaks_cross_validated (DataFrame): The list of peaks and troughs after cross-validation.

    PROPERTIES
        waves (DataFrame): An alias for peaks_cross_validated if calculated, else peaks_sub_c, for better access to the final results.
            Index: RangeIndex
            Columns:
                location: The index of the peak or trough within raw_data.
                y_position: The value of the peak or trough within raw_data.
                prominence: The prominence of the peak or trough (as calculated with respect to other peaks and troughs, not with resepct to all of raw_data).
                peak_ind: 0 for a trough, 1 for a peak.

    METHODS
        __init__: After setting the parameters, this calls run to immediately execute the algorithm.
        waves: Gets the most recent list of peaks and troughs
        run: Finds the list of peaks and troughs in raw_data, then calls the Sub-Algorithms A, B, C and D to find the
        waves.
        cross_validate: Imputes the presence of additional waves in the from those in a second wavelist.
    """

    def __init__(self, raw_data: Series, series_name: str,
                 t_sep_a: int, prominence_threshold: float, prominence_height_threshold: float):
        """
        Creates the WaveList object and calls run to find the waves using the set parameters

        Parameters:
            raw_data (Series): The original data from which the peaks and troughs are identified.
            series_name (str): The name of the series, for labelling plots.
            t_sep_a (int): Threshold specifying minimum wave duration.
            prominence_threshold (float): The minimum prominence which a wave must have.
            proportional_prominence_threshold (float): The minimum prominence which a peak must have, as a ratio of the
            value at the peak.
        """

        # input data
        self.raw_data = raw_data
        self.series_name = series_name

        # configuration parameters
        self.t_sep_a = t_sep_a
        self.prominence_threshold = prominence_threshold
        self.prominence_height_threshold = prominence_height_threshold

        # peaks and troughs of waves are calculated by run()
        self.peaks_initial, self.peaks_sub_a, self.peaks_sub_b, self.peaks_sub_c = self.run()
        self.peaks_cross_validated = None

    @property
    def waves(self):
        """ Provides the list of waves, peaks_sub_c or peaks_cross_validated"""
        if isinstance(self.peaks_cross_validated, DataFrame):
            return self.peaks_cross_validated
        else:
            return self.peaks_sub_c

    def run(self) -> (DataFrame, DataFrame, DataFrame, DataFrame):
        """ Executes the algorithm by finding the initial list of peaks and troughs, then calling A through D. """

        peaks_initial, prominence_updater = algorithm_init.run(self.raw_data)

        peaks_sub_a = algorithm_a.run(
            input_data_df=peaks_initial,
            prominence_updater=prominence_updater,
            t_sep_a=self.t_sep_a)

        peaks_sub_b = algorithm_b.run(
            raw_data=self.raw_data,
            input_data_df=peaks_sub_a,
            prominence_updater=prominence_updater,
            t_sep_a=self.t_sep_a)

        peaks_sub_c = algorithm_c_and_d.run(
            raw_data=self.raw_data,
            input_data_df=peaks_sub_b,
            prominence_threshold=self.prominence_threshold,
            proportional_prominence_threshold=self.prominence_height_threshold)

        return peaks_initial, peaks_sub_a, peaks_sub_b, peaks_sub_c

    def cross_validate(self, reference_wavelist: WaveList, plot: bool = False,
            plot_path: str = '', title: str = '') -> DataFrame:
        """
        Imputes the presence of additional waves in the from those in a reference_wavelist and stores them in peaks_cross_validated.

        Parameters:
            reference_wavelist (WaveList): The WaveList object which will be used to impute the additional waves.
            plot (bool): Whether the output should be plotted.
            plot_path (str): Location to store plots.
            title (str): Title for plots

        Returns:
            cross_validate(reference_wavelist, plot, plot_path, title): A DataFrame with the peaks and troughs after
            cross-validation against reference_wavelist.
        """

        # use the plotting tools - import now to avoid circularity
        import wavefinder.waveplotter as waveplotter

        # extract relevant data and parameters from the WaveList objects
        prominence_threshold = self.prominence_threshold
        proportional_prominence_threshold = self.prominence_height_threshold

        input_data = self.raw_data
        input_sub_b = self.peaks_sub_b
        input_sub_c = self.peaks_sub_c

        reference_sub_c = reference_wavelist.peaks_sub_c
        reference_peaks = reference_sub_c[reference_sub_c.peak_ind == 1].location
        reference_troughs = reference_sub_c[reference_sub_c.peak_ind == 0].location

        results = input_sub_c.copy()

        # iterate through the waves in the reference time series
        for i, reference_peak in enumerate(reference_peaks):
            # identify the start and end of the wave
            window_start = reference_troughs.iloc[i - 1] if i > 0 else 0
            window_end = reference_troughs.iloc[i] if i < len(reference_troughs) else input_data.index[-1]
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
        results = trough_finder.run(results, result_troughs, input_data, prominence_threshold,
                                    proportional_prominence_threshold)


        # plot the results if required
        if plot:
            waveplotter.plot_cross_validator(self, reference_wavelist, results, title, plot_path)

        self.peaks_cross_validated = results

        return results