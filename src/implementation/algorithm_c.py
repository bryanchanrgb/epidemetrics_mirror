from pandas import DataFrame
import matplotlib.pyplot as plt
from implementation.config import Config
from data_provider import DataProvider


class AlgorithmC:
    def __init__(self, config: Config, data_provider: DataProvider) -> DataFrame:
        self.config = config
        self.data_provider = data_provider

    @staticmethod
    def apply(data: DataFrame, sub_b: DataFrame, population: int, field: str, config: Config) -> DataFrame:

        if field == 'dead_per_day_smooth':
            abs_prominence_threshold = config.abs_prominence_threshold_dead
            rel_prominence_threshold = config.rel_prominence_threshold_dead
            rel_prominence_max_threshold = config.rel_prominence_max_threshold_dead
        else:
            abs_prominence_threshold = config.abs_prominence_threshold
            rel_prominence_threshold = config.rel_prominence_threshold
            rel_prominence_max_threshold = config.rel_prominence_max_threshold

        # prominence filter will use the larger of the absolute prominence threshold and relative prominence threshold
        # we cap the relative prominence threshold to rel_prominence_max_threshold
        prominence_threshold = max(abs_prominence_threshold,
                                   min(rel_prominence_threshold * population / config.rel_to_constant,
                                       rel_prominence_max_threshold))
        results = sub_b.copy()
        results = results.sort_values(by='location').reset_index(drop=True)
        # filter out troughs and peaks below prominence threshold
        result_peaks = results[results['peak_ind'] == 1].reset_index(drop=True)
        result_troughs = results[results['peak_ind'] == 0].reset_index(drop=True)

        # filter out a peak and its corresponding trough if the peak does not meet the prominence threshold
        result_peaks_c = result_peaks[result_peaks['prominence'] >= prominence_threshold]
        # filter out relatively low prominent peaks
        result_peaks_d = result_peaks_c[
            (result_peaks_c['prominence'] >= config.prominence_height_threshold * result_peaks_c['y_position'])]
        # between each remaining peak, retain the trough with the lowest value
        result_peaks_d = result_peaks_d.reset_index(drop=True)
        results = result_peaks_d
        for i in result_peaks_d.index:
            if i < max(result_peaks_d.index):
                candidate_troughs = result_troughs[(result_troughs['location'] >= result_peaks_d.loc[i, 'location']) &
                                                   (result_troughs['location'] <= result_peaks_d.loc[
                                                       i + 1, 'location'])]
                if len(candidate_troughs) > 0:
                    candidate_troughs = candidate_troughs.loc[candidate_troughs.idxmin()['y_position']]
                    results = results.append(candidate_troughs, ignore_index=True)
        results = results.sort_values(by='location').reset_index(drop=True)

        # add final trough after final peak
        if len(result_peaks_d) > 0:
            candidate_troughs = result_troughs[result_troughs.location >= result_peaks_d.location.iloc[-1]]
            if len(candidate_troughs) > 0:
                candidate_troughs = candidate_troughs.loc[candidate_troughs.idxmin()['y_position']]
                final_maximum = max(data[(data.index > candidate_troughs.location)][field])
                if (candidate_troughs.y_position <= (1 - config.prominence_height_threshold) *
                    result_peaks_d.y_position.iloc[-1]) and (
                        result_peaks_d.y_position.iloc[-1] - candidate_troughs.y_position >= prominence_threshold):
                    if (candidate_troughs.y_position <= (
                            1 - config.prominence_height_threshold) * final_maximum) and (
                            final_maximum - candidate_troughs.y_position >= prominence_threshold):
                        results = results.append(candidate_troughs, ignore_index=True)

        return results

    def run(self, sub_a: DataFrame, sub_b: DataFrame, country: str, field: str = 'new_per_day_smooth',
            plot: bool = False) -> DataFrame:
        data = self.data_provider.get_series(country, field)
        population = self.data_provider.get_population(country)
        results = self.apply(data, sub_b, population, field, self.config)
        if plot:
            self.plot(data, sub_a, sub_b, results, field)
        return results

    def plot(self, data: DataFrame, sub_a: DataFrame, sub_b: DataFrame, results: DataFrame, field: str):
        fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
        # plot peaks-trough pairs from sub_a
        ax0.set_title('After Sub Algorithm A')
        ax0.plot(data[field].values)
        ax0.scatter(sub_a['location'].values,
                    data[field].values[sub_a['location'].values.astype(int)], color='red', marker='o')
        # plot peaks-trough pairs from sub_b
        ax1.set_title('After Sub Algorithm B')
        ax1.plot(data[field].values)
        ax1.scatter(sub_b['location'].values,
                    data[field].values[sub_b['location'].values.astype(int)], color='red', marker='o')
        # plot peaks from sub_c
        ax2.set_title('After Sub Algorithm C & D')
        ax2.plot(data[field].values)
        ax2.scatter(results['location'].values,
                    data[field].values[results['location'].values.astype(int)], color='red', marker='o')
        plt.show()
