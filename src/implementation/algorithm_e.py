import os
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from implementation.config import Config
from data_provider import DataProvider
from implementation.trough_finder import TroughFinder


class AlgorithmE:
    def __init__(self, config: Config, data_provider: DataProvider) -> DataFrame:
        self.config = config
        self.data_provider = data_provider

    @staticmethod
    def apply(data: DataFrame, cases_sub_b: DataFrame, cases_sub_c: DataFrame,
              deaths_sub_c: DataFrame, population: int, config: Config) -> DataFrame:

        # set up prominence thresholds
        field = 'new_per_day_smooth'
        abs_prominence_threshold = config.prominence_thresholds(field)['abs_prominence_threshold']
        rel_prominence_threshold = config.prominence_thresholds(field)['rel_prominence_threshold']
        rel_prominence_max_threshold = config.prominence_thresholds(field)['rel_prominence_max_threshold']
        prominence_height_threshold = config.prominence_thresholds(field)['prominence_height_threshold']
        prominence_threshold = max(abs_prominence_threshold,
                                   min(rel_prominence_threshold * population / config.rel_to_constant,
                                       rel_prominence_max_threshold))

        # when a wave of deaths ends, check if there is match between the number of peaks in both series.
        # if not, then add the most prominent peak for the output of sub-algorithm B in the self.d_match days before or after

        death_peaks = deaths_sub_c[deaths_sub_c.peak_ind == 1].location
        death_troughs = deaths_sub_c[deaths_sub_c.peak_ind == 0].location
        results = cases_sub_c.copy()

        # iterate through the waves of deaths
        for i, death_peak in enumerate(death_peaks):
            # identify the start and end of the wave
            window_start = death_troughs.iloc[i - 1] if i > 0 else 0
            window_end = death_troughs.iloc[i] if i < len(death_troughs) else data.index[-1]
            # check if a case peak exists during the wave - if it does then continue
            if np.any([True if (x >= window_start) and (x <= window_end)
                       else False for x in cases_sub_c.loc[cases_sub_c['peak_ind'] == 1, 'location']]):
                continue
            # if peak in cases_sub_b output use the highest one
            candidates = cases_sub_b[(cases_sub_b['peak_ind'] == 1) &
                                     (cases_sub_b['location'] >= window_start) & (
                                             cases_sub_b['location'] <= window_end)]
            if len(candidates) > 0:
                new_peak = candidates.loc[candidates.idxmax()['y_position']]
                results = results.append(new_peak)

        # next add back any troughs as in algorithm_c
        result_troughs = cases_sub_b[cases_sub_b.peak_ind == 0]
        results = results[results.peak_ind == 1]
        results = TroughFinder.run(results,result_troughs,data,'new_per_day_smooth',prominence_threshold, prominence_height_threshold)

        return results

    def run(self, cases_sub_b: DataFrame, cases_sub_c: DataFrame, deaths_sub_c: DataFrame, country: str,
            plot: bool = False) -> DataFrame:
        data = self.data_provider.get_series(country, field='new_per_day_smooth')
        deaths_data = self.data_provider.get_series(country, field='dead_per_day_smooth')
        population = self.data_provider.get_population(country)
        results = self.apply(data, cases_sub_b, cases_sub_c, deaths_sub_c, population, self.config)
        if plot:
            self.plot(data, country, cases_sub_c, results, deaths_data, deaths_sub_c)
        return results

    def plot(self, data: DataFrame, country: str, cases_sub_c: DataFrame, results: DataFrame,
             deaths_data: DataFrame, deaths_sub_c: DataFrame):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        # plot peaks after sub_c
        axs[0, 0].set_title('After Sub Algorithm C & D')
        axs[0, 0].plot(data['new_per_day_smooth'].values)
        axs[0, 0].scatter(cases_sub_c['location'].values,
                          data['new_per_day_smooth'].values[
                              cases_sub_c['location'].values.astype(int)], color='red', marker='o')
        # plot peaks from sub_e
        axs[0, 1].set_title('After Sub Algorithm E')
        axs[0, 1].plot(data['new_per_day_smooth'].values)
        axs[0, 1].scatter(results['location'].values,
                          data['new_per_day_smooth'].values[
                              results['location'].values.astype(int)], color='red', marker='o')
        # plot death peaks
        axs[1, 1].set_title('Death Peaks')
        axs[1, 1].plot(deaths_data['dead_per_day_smooth'].values)
        axs[1, 1].scatter(deaths_sub_c['location'].values,
                          deaths_data['dead_per_day_smooth'].values[
                              deaths_sub_c['location'].values.astype(int)], color='red', marker='o')

        fig.tight_layout()
        plt.savefig(os.path.join(self.config.plot_path, country + '_algorithm_e.png'))
        plt.close('all')
