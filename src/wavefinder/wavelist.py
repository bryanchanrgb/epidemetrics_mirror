from pandas import DataFrame, Series

import wavefinder.subalgorithms.algorithm_init as algorithm_init
import wavefinder.subalgorithms.algorithm_a as algorithm_a
import wavefinder.subalgorithms.algorithm_b as algorithm_b
import wavefinder.subalgorithms.algorithm_c as algorithm_c


class WaveList:
    def __init__(self, raw_data: Series, series_name: str,
                 t_sep_a: int, prominence_threshold: float, prominence_height_threshold: float):
        # input data
        self.raw_data = raw_data
        self.series_name = series_name

        # configuration parameters
        self.t_sep_a = t_sep_a
        self.prominence_threshold = prominence_threshold
        self.prominence_height_threshold = prominence_height_threshold

        # peaks and troughs of waves are calculated by run()
        self.peaks_initial, self.peaks_sub_a, self.peaks_sub_b, self.peaks_sub_c = self.run()

    def run(self) -> (DataFrame, DataFrame, DataFrame, DataFrame):
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

        peaks_sub_c = algorithm_c.run(
            raw_data=self.raw_data,
            input_data_df=peaks_sub_b,
            prominence_threshold=self.prominence_threshold,
            proportional_prominence_threshold=self.prominence_height_threshold)

        return peaks_initial, peaks_sub_a, peaks_sub_b, peaks_sub_c
