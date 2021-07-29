from pandas import DataFrame, Series

import wavefinder.subalgorithms.algorithm_init as algorithm_init
import wavefinder.subalgorithms.algorithm_a as algorithm_a
import wavefinder.subalgorithms.algorithm_b as algorithm_b
import wavefinder.subalgorithms.algorithm_c_and_d as algorithm_c_and_d


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
        peaks_sub_a: The list of peaks and troughs after Sub-Algorithm A has merged short waves.
        peaks_sub_b: The list of peaks and troughs after Sub-Algorithm B has merged short transient features.
        peaks_sub_c: The list of peaks and troughs after Sub-Algorithms C and D have merged less prominent waves.
        waves: An alias for peaks_sub_c, for better readability outside the package.

    METHODS
        __init__: After setting the parameters, this calls run to immediately execute the algorithm.
        run: Finds the list of peaks and troughs in raw_data, then calls the Sub-Algorithms A, B, C and D to find the
        waves.
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

        # provide an alias for better readability outside the package
        self.waves = self.peaks_sub_c

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
