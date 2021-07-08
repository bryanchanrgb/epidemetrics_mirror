import os
import numpy as np
from pandas import DataFrame
from typing import Dict
import matplotlib.pyplot as plt
from implementation.config import Config
from data_provider import DataProvider
from implementation.trough_finder import TroughFinder
from plot_helper import plot_e


class AlgorithmE:
    def __init__(self, config: Config, data_provider: DataProvider, country: str) -> DataFrame:
        self.config = config
        self.country = country
        self.data_provider = data_provider
        self.params = config.prominence_thresholds('new_per_day_smooth')
        self.params['rel_to_constant'] = config.rel_to_constant

    @staticmethod
    def apply(data: DataFrame, cases_sub_b: DataFrame, cases_sub_c: DataFrame,
              deaths_sub_c: DataFrame, population: int, params: Dict) -> DataFrame:

        # set up prominence thresholds
        prominence_threshold = max(params['abs_prominence_threshold'],
                                   min(params['rel_prominence_threshold'] * population / params['rel_to_constant'],
                                       params['rel_prominence_max_threshold']))

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
        raw_data = data['new_per_day_smooth']
        results = TroughFinder.run(results, result_troughs, raw_data, prominence_threshold,
                                   params['prominence_height_threshold'])

        return results

    def run(self, cases_sub_b: DataFrame, cases_sub_c: DataFrame, deaths_sub_c: DataFrame,
            plot: bool = False) -> DataFrame:
        data = self.data_provider.get_series(self.country, field='new_per_day_smooth')
        deaths_data = self.data_provider.get_series(self.country, field='dead_per_day_smooth')
        population = self.data_provider.get_population(self.country)
        results = self.apply(data, cases_sub_b, cases_sub_c, deaths_sub_c, population, self.params)
        if plot:
            plot_e(data, self.country, cases_sub_c, results, deaths_data, deaths_sub_c, self.config.plot_path)
        return results
