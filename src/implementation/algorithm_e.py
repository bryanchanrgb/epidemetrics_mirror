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
        # basically look into the output of sub algorithm b for cases and sub algorithm c for deaths
        # if there is a case peak in self.d_match days before ignore
        # else use the most prominent peak in the d days before
        # if the first death peak is before d_match then no use of doing this
        if deaths_sub_c['location'].min() < self.config.d_match:
            return cases_sub_c

        data = self.data_provider.get_series(country, field='new_per_day_smooth')
        results = cases_sub_c.copy()

        for i, death_peak in enumerate(deaths_sub_c['location']):
            # check if for this death peak there is a peak in cases between (death_peak - d_match) and death_peak
            # if peak already there continue
            if np.any([True if (x >= death_peak - self.config.d_match) and (x <= death_peak)
                       else False for x in cases_sub_c.loc[cases_sub_c['peak_ind'] == 1, 'location']]):
                continue
            # if peak in cases_sub_b output use the most prominent one
            elif np.any([True if (x >= death_peak - self.config.d_match) and (x <= death_peak)
                         else False for x in cases_sub_b.loc[cases_sub_b['peak_ind'] == 1, 'location']]):
                # potential candidates for peaks are those within range in cases_sub_b
                candidates = cases_sub_b[(cases_sub_b['peak_ind'] == 1) &
                                         (cases_sub_b['location'] >= death_peak - self.config.d_match) & (
                                                 cases_sub_b['location'] <= death_peak)]
                results = results.append(candidates.loc[candidates.idxmax()['prominence']])
                continue
            # if nothing, could use max - but might violate t_sep rule...
            else:
                continue
        results = results.sort_values(by=['location'])

        if plot:
            self.plot(data, country, cases_sub_a, cases_sub_b, cases_sub_c, results)

        return results

    def plot(self, data, country, cases_sub_a, cases_sub_b, cases_sub_c, results):
        fig, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=4)
        # plot peaks-trough pairs from sub_a
        ax0.set_title('After Sub Algorithm A')
        ax0.plot(data['new_per_day_smooth'].values)
        ax0.scatter(cases_sub_a['location'].values,
                    data['new_per_day_smooth'].values[
                        cases_sub_a['location'].values.astype(int)], color='red', marker='o')
        # plot peaks-trough pairs from sub_b
        ax1.set_title('After Sub Algorithm B')
        ax1.plot(data['new_per_day_smooth'].values)
        ax1.scatter(cases_sub_b['location'].values,
                    data['new_per_day_smooth'].values[
                        cases_sub_b['location'].values.astype(int)], color='red', marker='o')
        # plot peaks after sub_c
        ax2.set_title('After Sub Algorithm C & D')
        ax2.plot(data['new_per_day_smooth'].values)
        ax2.scatter(cases_sub_c['location'].values,
                    data['new_per_day_smooth'].values[
                        cases_sub_c['location'].values.astype(int)], color='red', marker='o')
        # plot peaks from sub_e
        ax3.set_title('After Sub Algorithm E')
        ax3.plot(data['new_per_day_smooth'].values)
        ax3.scatter(results['location'].values,
                    data['new_per_day_smooth'].values[
                        results['location'].values.astype(int)], color='red', marker='o')

        fig.tight_layout()
        plt.savefig(os.path.join(self.config.plot_path, country + '_algorithm_e.png'))
        plt.close('all')
