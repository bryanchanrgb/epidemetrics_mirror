from pandas import DataFrame
import matplotlib.pyplot as plt
from implementation.config import Config
from data_provider import DataProvider
from implementation.trough_finder import TroughFinder


class AlgorithmC:
    def __init__(self, config: Config, data_provider: DataProvider) -> DataFrame:
        self.config = config
        self.data_provider = data_provider

    @staticmethod
    def apply(data: DataFrame, input_data_df: DataFrame, population: int, field: str, config: Config) -> DataFrame:

        abs_prominence_threshold = config.prominence_thresholds(field)['abs_prominence_threshold']
        rel_prominence_threshold = config.prominence_thresholds(field)['rel_prominence_threshold']
        rel_prominence_max_threshold = config.prominence_thresholds(field)['rel_prominence_max_threshold']
        prominence_height_threshold = config.prominence_thresholds(field)['prominence_height_threshold']

        # prominence filter will use the larger of the absolute prominence threshold and relative prominence threshold
        # we cap the relative prominence threshold to rel_prominence_max_threshold
        prominence_threshold = max(abs_prominence_threshold,
                                   min(rel_prominence_threshold * population / config.rel_to_constant,
                                       rel_prominence_max_threshold))
        df = input_data_df.copy()
        df = df.sort_values(by='location').reset_index(drop=True)
        # filter out troughs and peaks below prominence threshold
        peaks = df[df['peak_ind'] == 1].reset_index(drop=True)
        troughs = df[df['peak_ind'] == 0].reset_index(drop=True)

        # filter out a peak and its corresponding trough if the peak does not meet the prominence threshold
        peaks_c = peaks[peaks['prominence'] >= prominence_threshold]
        # filter out relatively low prominent peaks
        peaks_d = peaks_c[
            (peaks_c['prominence'] >= prominence_height_threshold * peaks_c['y_position'])]
        # between each remaining peak, retain the trough with the lowest value
        df = TroughFinder.run(peaks_d,troughs,data,field,prominence_threshold, prominence_height_threshold)

        return df

    def run(self, input_data_df: DataFrame, country: str, field: str = 'new_per_day_smooth',
            plot: bool = False) -> DataFrame:
        data = self.data_provider.get_series(country, field)
        population = self.data_provider.get_population(country)
        output_data_df = self.apply(data, input_data_df, population, field, self.config)
        if plot:
            self.plot(data, input_data_df, output_data_df, field)
        return output_data_df

    def plot(self, data: DataFrame, after_sub_b: DataFrame, after_sub_c: DataFrame, field: str):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
        ax1.set_title('After Sub Algorithm B')
        ax1.plot(data[field].values)
        ax1.scatter(after_sub_b['location'].values,
                    data[field].values[after_sub_b['location'].values.astype(int)], color='red', marker='o')
        # plot peaks from sub_c
        ax2.set_title('After Sub Algorithm C & D')
        ax2.plot(data[field].values)
        ax2.scatter(after_sub_c['location'].values,
                    data[field].values[after_sub_c['location'].values.astype(int)], color='red', marker='o')
        plt.show()
