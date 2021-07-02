import os
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from implementation.config import Config
from data_provider import DataProvider


class AlgorithmE:
    def __init__(self, config: Config, data_provider: DataProvider) -> DataFrame:
        self.config = config
        self.data_provider = data_provider

    def run(self, cases_sub_a, cases_sub_b, cases_sub_c, deaths_sub_c, country, plot=False):
        # when a wave of deaths ends, check if there is match between the number of peaks in both series.
        # if not, then add the most prominent peak for the output of sub-algorithm B in the self.d_match days before or after

        data = self.data_provider.get_series(country, field='new_per_day_smooth')
        deaths_data = self.data_provider.get_series(country, field='dead_per_day_smooth')

        death_peaks = deaths_sub_c[deaths_sub_c.peak_ind == 1].location
        death_troughs = deaths_sub_c[deaths_sub_c.peak_ind == 0].location
        results = cases_sub_c.copy()

        # iterate through the waves of deaths
        for i, death_peak in enumerate(death_peaks):
            # identify the start and end of the wave
            window_start = death_troughs.iloc[i-1] if i>0 else 0
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
        results = results[results.peak_ind==1].reset_index(drop=True)
        for i in results.index:
            if i < max(results.index):
                candidate_troughs = result_troughs[(result_troughs['location'] >= results.loc[i, 'location']) &
                                                   (result_troughs['location'] <= results.loc[
                                                       i + 1, 'location'])]
                if len(candidate_troughs) > 0:
                    candidate_troughs = candidate_troughs.loc[candidate_troughs.idxmin()['y_position']]
                    results = results.append(candidate_troughs, ignore_index=True)
        results = results.sort_values(by='location').reset_index(drop=True)

        # add final trough after final peak
        if len(results) > 0:
            # obtain prominence threshold
            population = \
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]['value'].values[0]
            abs_prominence_threshold = self.config.abs_prominence_threshold
            rel_prominence_threshold = self.config.rel_prominence_threshold
            rel_prominence_max_threshold = self.config.rel_prominence_max_threshold
            prominence_threshold = max(abs_prominence_threshold,
                                       min(rel_prominence_threshold * population / self.config.rel_to_constant,
                                           rel_prominence_max_threshold))

            candidate_troughs = result_troughs[result_troughs.location >= results.location.iloc[-1]]
            if len(candidate_troughs) > 0:
                candidate_troughs = candidate_troughs.loc[candidate_troughs.idxmin()['y_position']]
                final_maximum = max(data[(data.index > candidate_troughs.location)]['new_per_day_smooth'])
                if (candidate_troughs.y_position <= (1 - self.config.prominence_height_threshold) *
                    results.y_position.iloc[-1]) and (
                        results.y_position.iloc[-1] - candidate_troughs.y_position >= prominence_threshold):
                    if (candidate_troughs.y_position <= (
                            1 - self.config.prominence_height_threshold) * final_maximum) and (
                            final_maximum - candidate_troughs.y_position >= prominence_threshold):
                        results = results.append(candidate_troughs, ignore_index=True)

        if plot:
            self.plot(data, country, cases_sub_a, cases_sub_b, cases_sub_c, results, deaths_data, deaths_sub_c)

        return results

    def plot(self, data, country, cases_sub_a, cases_sub_b, cases_sub_c, results, deaths_data, deaths_sub_c):
        fig, axs = plt.subplots(nrows=2, ncols=2)
        # plot peaks after sub_c
        axs[0,0].set_title('After Sub Algorithm C & D')
        axs[0,0].plot(data['new_per_day_smooth'].values)
        axs[0,0].scatter(cases_sub_c['location'].values,
                    data['new_per_day_smooth'].values[
                        cases_sub_c['location'].values.astype(int)], color='red', marker='o')
        # plot peaks from sub_e
        axs[0,1].set_title('After Sub Algorithm E')
        axs[0,1].plot(data['new_per_day_smooth'].values)
        axs[0,1].scatter(results['location'].values,
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
