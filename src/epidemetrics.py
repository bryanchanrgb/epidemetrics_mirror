import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import datetime
from pandas import DataFrame
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from data_provider import DataProvider
from implementation.algorithm_a import AlgorithmA
from implementation.algorithm_b import AlgorithmB
from implementation.algorithm_c import AlgorithmC
from implementation.algorithm_e import AlgorithmE
from implementation.config import Config


class Epidemetrics:
    def __init__(self, config: Config, data_provider: DataProvider, plot_path: str):
        self.config = config
        self.data_provider = data_provider
        self.plot_path = plot_path
        self.prepare_output_dirs(self.plot_path)

        self.algorithm_a = AlgorithmA(self.config, self.data_provider)
        self.algorithm_b = AlgorithmB(self.config, self.data_provider, algorithm_a=self.algorithm_a)
        self.algorithm_c = AlgorithmC(self.config, self.data_provider)
        self.algorithm_e = AlgorithmE(self.config, self.data_provider)

    def prepare_output_dirs(self, path: str):
        print(f'Preparing output directory: {path}')
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error: {path}: {e.strerror}")
        Path(path).mkdir(parents=True, exist_ok=True)

    def epi_find_peaks(self, country: str, plot: bool = False, save: bool = False) -> DataFrame:
        # match parameter tries to use death waves to detect case waves under sub_algorithm_e
        cases = self.data_provider.get_series(country=country, field='new_per_day_smooth')
        if len(cases) == 0:
            raise ValueError
        cases_sub_a = self.algorithm_a.run(country=country, field='new_per_day_smooth')
        cases_sub_b = self.algorithm_b.run(cases_sub_a, country=country, field='new_per_day_smooth')
        cases_sub_c = self.algorithm_c.run(
            sub_a=cases_sub_a, sub_b=cases_sub_b, country=country, field='new_per_day_smooth')
        # compute equivalent series for deaths
        deaths = self.data_provider.get_series(country=country, field='dead_per_day_smooth')
        if len(deaths) == 0:
            raise ValueError
        deaths_sub_a = self.algorithm_a.run(country=country, field='dead_per_day_smooth')
        deaths_sub_b = self.algorithm_b.run(deaths_sub_a, country=country, field='dead_per_day_smooth')
        deaths_sub_c = self.algorithm_c.run(
            sub_a=deaths_sub_a, sub_b=deaths_sub_b, country=country, field='dead_per_day_smooth')
        # run sub algorithm e
        cases_sub_e = self.algorithm_e.run(cases_sub_a, cases_sub_b, cases_sub_c, deaths_sub_c, country=country)
        # compute plots
        if plot:
            self.plot_peaks(cases, deaths, country, cases_sub_a, cases_sub_b, cases_sub_c,
                            deaths_sub_a, deaths_sub_b, deaths_sub_c, save)

        return cases_sub_e

    def plot_peaks(self, cases, deaths, country, cases_sub_a, cases_sub_b, cases_sub_c,
                   deaths_sub_a, deaths_sub_b, deaths_sub_c, save):

        peak = find_peaks(cases['new_per_day_smooth'].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in cases['new_per_day_smooth'].values], prominence=0, distance=1)
        cases_pre_algo = pd.DataFrame(data=np.transpose(np.append(cases.index[peak[0]], cases.index[trough[0]])),
                                      columns=['location'])
        peak = find_peaks(deaths['dead_per_day_smooth'].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in deaths['dead_per_day_smooth'].values], prominence=0, distance=1)
        deaths_pre_algo = pd.DataFrame(data=np.transpose(np.append(deaths.index[peak[0]], deaths.index[trough[0]])),
                                       columns=['location'])

        fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(14, 7))
        plt.suptitle(country)

        axs[0, 0].set_title('Cases Before Algorithm')
        axs[0, 0].plot(cases['new_per_day_smooth'].values)
        axs[0, 0].scatter(cases_pre_algo['location'].values,
                          cases['new_per_day_smooth'].values[
                              cases_pre_algo['location'].values.astype(int)], color='red', marker='o')
        axs[0, 0].get_xaxis().set_visible(False)
        axs[0, 0].get_yaxis().set_visible(False)

        axs[0, 1].set_title('Cases After Sub Algorithm A')
        axs[0, 1].plot(cases['new_per_day_smooth'].values)
        axs[0, 1].scatter(cases_sub_a['location'].values,
                          cases['new_per_day_smooth'].values[
                              cases_sub_a['location'].values.astype(int)], color='red', marker='o')
        axs[0, 1].get_xaxis().set_visible(False)
        axs[0, 1].get_yaxis().set_visible(False)

        axs[0, 2].set_title('Cases After Sub Algorithm B')
        axs[0, 2].plot(cases['new_per_day_smooth'].values)
        axs[0, 2].scatter(cases_sub_b['location'].values,
                          cases['new_per_day_smooth'].values[
                              cases_sub_b['location'].values.astype(int)], color='red', marker='o')
        axs[0, 2].get_xaxis().set_visible(False)
        axs[0, 2].get_yaxis().set_visible(False)

        axs[0, 3].set_title('Cases After Sub Algorithm C&D')
        axs[0, 3].plot(cases['new_per_day_smooth'].values)
        axs[0, 3].scatter(cases_sub_c['location'].values,
                          cases['new_per_day_smooth'].values[
                              cases_sub_c['location'].values.astype(int)], color='red', marker='o')
        axs[0, 3].get_xaxis().set_visible(False)
        axs[0, 3].get_yaxis().set_visible(False)

        axs[1, 0].set_title('Deaths Before Algorithm')
        axs[1, 0].plot(deaths['dead_per_day_smooth'].values)
        axs[1, 0].scatter(deaths_pre_algo['location'].values,
                          deaths['dead_per_day_smooth'].values[
                              deaths_pre_algo['location'].values.astype(int)], color='red', marker='o')
        axs[1, 0].get_xaxis().set_visible(False)
        axs[1, 0].get_yaxis().set_visible(False)

        axs[1, 1].set_title('Deaths After Sub Algorithm A')
        axs[1, 1].plot(deaths['dead_per_day_smooth'].values)
        axs[1, 1].scatter(deaths_sub_a['location'].values,
                          deaths['dead_per_day_smooth'].values[
                              deaths_sub_a['location'].values.astype(int)], color='red', marker='o')
        axs[1, 1].get_xaxis().set_visible(False)
        axs[1, 1].get_yaxis().set_visible(False)

        axs[1, 2].set_title('Deaths After Sub Algorithm B')
        axs[1, 2].plot(deaths['dead_per_day_smooth'].values)
        axs[1, 2].scatter(deaths_sub_b['location'].values,
                          deaths['dead_per_day_smooth'].values[
                              deaths_sub_b['location'].values.astype(int)], color='red', marker='o')
        axs[1, 2].get_xaxis().set_visible(False)
        axs[1, 2].get_yaxis().set_visible(False)

        axs[1, 3].set_title('Deaths After Sub Algorithm C&D')
        axs[1, 3].plot(deaths['dead_per_day_smooth'].values)
        axs[1, 3].scatter(deaths_sub_c['location'].values,
                          deaths['dead_per_day_smooth'].values[
                              deaths_sub_c['location'].values.astype(int)], color='red', marker='o')
        axs[1, 3].get_xaxis().set_visible(False)
        axs[1, 3].get_yaxis().set_visible(False)

        fig.tight_layout()

        if save:
            plt.savefig(os.path.join(self.plot_path, country + '.png'))
            plt.close('all')

    def _debug_case_death_ascertainment_plot(self):
        n = 50
        data = self.data_provider.epidemiology_series[
            ['countrycode', 'days_since_t0', 'date', 'confirmed', 'case_death_ascertainment']]
        worst_n_countries = pd.DataFrame.from_records(
            [{'countrycode': i, 'confirmed': data[data['countrycode'] == i]['confirmed'].iloc[-1]}
             for i in data['countrycode'].unique()]).sort_values(
            by=['confirmed'], ascending=False).iloc[0:n]['countrycode'].values
        data_t0 = data[(data['days_since_t0'] >= 0) & (data['countrycode'].isin(worst_n_countries))]
        data_date = data[(data['date'] >= datetime.date(2020, 3, 1)) & (data['countrycode'].isin(worst_n_countries))]
        fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
        for country in worst_n_countries:
            if country in self.config.debug_countries_of_interest:
                ax[0].plot(data_t0[data_t0['countrycode'] == country]['days_since_t0'].values,
                           data_t0[data_t0['countrycode'] == country]['case_death_ascertainment'].values,
                           linewidth=2, label=country)
                ax[1].plot(data_date[data_date['countrycode'] == country]['date'].values,
                           data_date[data_date['countrycode'] == country]['case_death_ascertainment'].values,
                           linewidth=2, label=country)
            else:
                ax[0].plot(data_t0[data_t0['countrycode'] == country]['days_since_t0'].values,
                           data_t0[data_t0['countrycode'] == country]['case_death_ascertainment'].values, color='grey',
                           alpha=0.25, linewidth=1, label='_nolegend_')
                ax[1].plot(data_date[data_date['countrycode'] == country]['date'].values,
                           data_date[data_date['countrycode'] == country]['case_death_ascertainment'].values,
                           color='grey',
                           alpha=0.25, linewidth=1, label='_nolegend_')
            ax[0].legend()
            ax[0].set_xlabel('Days Since T0')
            ax[0].set_ylabel('Cases / Deaths (+9d)')
            ax[0].set_ylim([0, 1e2])
            ax[1].legend()
            ax[1].set_xlabel('Date')
            ax[1].set_ylabel('Cases / Deaths (+9d)')
            ax[1].set_ylim([0, 1e2])
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_path, 'inverse_cfr.png'))
        plt.close('all')
