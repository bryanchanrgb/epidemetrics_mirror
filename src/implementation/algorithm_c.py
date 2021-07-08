from pandas import DataFrame, Series
from typing import Dict
from implementation.config import Config
from data_provider import DataProvider
from implementation.trough_finder import TroughFinder


class AlgorithmC:
    def __init__(self, config: Config, data_provider: DataProvider, country: str,
                 field: str = 'new_per_day_smooth') -> DataFrame:
        self.config = config
        self.country = country
        self.data_provider = data_provider
        self.params = config.prominence_thresholds(field)
        self.params['rel_to_constant'] = config.rel_to_constant

    @staticmethod
    def apply(raw_data: Series, input_data_df: DataFrame, population: int, params: Dict) -> DataFrame:
        # prominence filter will use the larger of the absolute prominence threshold and relative prominence threshold
        # we cap the relative prominence threshold to rel_prominence_max_threshold
        prominence_threshold = max(params['abs_prominence_threshold'],
                                   min(params['rel_prominence_threshold'] * population / params['rel_to_constant'],
                                       params['rel_prominence_max_threshold']))
        df = input_data_df.copy()
        df = df.sort_values(by='location').reset_index(drop=True)
        # filter out troughs and peaks below prominence threshold
        peaks = df[df['peak_ind'] == 1].reset_index(drop=True)
        troughs = df[df['peak_ind'] == 0].reset_index(drop=True)

        # filter out a peak and its corresponding trough if the peak does not meet the prominence threshold
        peaks_c = peaks[peaks['prominence'] >= prominence_threshold]
        # filter out relatively low prominent peaks
        peaks_d = peaks_c[
            (peaks_c['prominence'] >= params['prominence_height_threshold'] * peaks_c['y_position'])]
        # between each remaining peak, retain the trough with the lowest value
        df = TroughFinder.run(peaks_d, troughs, raw_data, prominence_threshold, params['prominence_height_threshold'])

        return df

    def run(self, raw_data: Series, input_data_df: DataFrame) -> DataFrame:
        population = self.data_provider.get_population(self.country)
        output_data_df = self.apply(raw_data, input_data_df, population, self.params)
        return output_data_df
