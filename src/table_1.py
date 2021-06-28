import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import pingouin as pg
from epidemetrics import Epidemetrics
from data_provider import DataProvider
from implementation.algorithm_a import AlgorithmA
from implementation.algorithm_b import AlgorithmB
from implementation.algorithm_c import AlgorithmC
from implementation.config import Config


class Table_1:
    def __init__(self, config: Config, data_provider: DataProvider, epimetrics: Epidemetrics, data_path: str):
        self.config = config
        self.data_path = data_path
        self.epimetrics = epimetrics
        self.data_provider = data_provider

        self.algorithm_a = AlgorithmA(self.config, self.data_provider)
        self.algorithm_b = AlgorithmB(self.config, self.data_provider, algorithm_a=self.algorithm_a)
        self.algorithm_c = AlgorithmC(self.config, self.data_provider)

    def _classify(self, country, field='new_per_day_smooth'):
        data = self.data_provider.get_series(country=country, field=field)
        # class 0 reserved for misbehaving cases
        if (len(data) < 3) or \
                (country in self.config.exclude_countries) or \
                not (country in self.data_provider.wbi_table['countrycode'].values):
            return 0, None
        # method _find_peaks is only supported for new_cases_per_day as the cross-validation step requires death per day
        # for alternative fields the output of sub_algorithm_c is used
        if field == 'new_per_day_smooth':
            deaths = self.data_provider.get_series(country=country, field='dead_per_day_smooth')
            if len(deaths) < 3:
                return 0, None
            genuine_peaks = self.epimetrics.epi_find_peaks(country, plot=False, save=False)
            genuine_peaks = genuine_peaks[genuine_peaks['peak_ind'] == 1]
        else:
            sub_a = self.algorithm_a.run(country, field=field, plot=False)
            sub_b = self.algorithm_b.run(sub_a, country, field=field, plot=False)
            genuine_peaks = self.algorithm_c.run(sub_a, sub_b, country, field=field, plot=False)
            genuine_peaks = genuine_peaks[genuine_peaks['peak_ind'] == 1]

        population = \
            self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]['value'].values[0]
        if field == 'dead_per_day_smooth':
            abs_prominence_threshold = self.config.abs_prominence_threshold_dead
            rel_prominence_threshold = self.config.rel_prominence_threshold_dead
            rel_prominence_max_threshold = self.config.rel_prominence_max_threshold_dead
            class_1_threshold = self.config.class_1_threshold_dead
        else:
            abs_prominence_threshold = self.config.abs_prominence_threshold
            rel_prominence_threshold = self.config.rel_prominence_threshold
            rel_prominence_max_threshold = self.config.rel_prominence_max_threshold
            class_1_threshold = self.config.class_1_threshold

        # prominence filter will use the larger of the absolute prominence threshold and relative prominence threshold
        # we cap the relative prominence threshold to rel_prominence_max_threshold
        prominence_threshold = max(abs_prominence_threshold,
                                   min(rel_prominence_threshold * population / self.config.rel_to_constant,
                                       rel_prominence_max_threshold))

        peak_class = 2 * len(genuine_peaks)
        # if the last value is able to meet the constraints from sub algorithm C, we can
        if (peak_class > 0) and (genuine_peaks['location'].iloc[-1] < len(data)):
            last_peak_date = data['date'].values[int(genuine_peaks['location'].iloc[-1])]
            trough_value = min(data.loc[data['date'] > last_peak_date, 'new_per_day_smooth'])
            trough_date = data[data['date'] > last_peak_date]['date'].loc[
                int(data.loc[data['date'] > last_peak_date, 'new_per_day_smooth'].idxmin())]
            max_after_trough = np.nanmax(
                data.loc[data['date'] >= trough_date, 'new_per_day_smooth'])
            if (max_after_trough - trough_value >= prominence_threshold) \
                    and (
                    max_after_trough - trough_value >= self.config.prominence_height_threshold * max_after_trough):
                peak_class += 1
        elif (peak_class == 0) and not (country in self.config.exclude_countries):
            if np.nanmax(data['new_per_day_smooth']) >= class_1_threshold:
                peak_class += 1
        else:
            pass
        return peak_class, genuine_peaks

    # waiting implementation
    def _get_epi_panel(self):
        print('Preparing Epidemiological Results Table')
        '''epidemiology_static = pd.DataFrame(
            columns=['countrycode', 'country', 'class', 'population',
                     't0', 't0_relative', 't0_1_dead','t0_5_dead',
                     't0_10_dead', 'peak_1', 'peak_2', 'date_peak_1',
                     'date_peak_2', 'first_wave_start', 'first_wave_end', 'duration_first_wave',
                     'second_wave_start', 'second_wave_end','last_confirmed', 'last_dead',
                     'testing_available','peak_1_cfr','peak_2_cfr', 'dead_class','tests_class'])'''
        epidemiology_panel = pd.DataFrame()
        # wave parameters marked a w
        for country in tqdm(np.sort(self.data_provider.epidemiology_series['countrycode'].unique()),
                            desc='Preparing Epidemiological Results Table'):
            data = dict()
            data['countrycode'] = country
            data['country'] = np.nan
            data['class'] = np.nan
            data['class_coarse'] = np.nan  # one, two, three or more waves
            data['population'] = np.nan
            data['population_density'] = np.nan
            data['gni_per_capita'] = np.nan
            data['total_confirmed'] = np.nan
            data['total_dead'] = np.nan
            data['mortality_rate'] = np.nan
            data['case_rate'] = np.nan
            data['peak_case_rate'] = np.nan
            data['stringency_response_time'] = np.nan
            data['total_stringency'] = np.nan
            data['testing_response_time'] = np.nan  # days to reach 10 tests per rel_to
            data['t0'] = np.nan
            data['t0_relative'] = np.nan
            data['t0_1_dead'] = np.nan
            data['t0_5_dead'] = np.nan
            data['t0_10_dead'] = np.nan
            data['testing_available'] = np.nan
            data['rel_to_constant'] = self.config.rel_to_constant
            data['peak_1'] = np.nan  # w
            data['peak_1_per_rel_to'] = np.nan  # w
            data['date_peak_1'] = np.nan  # w
            data['wave_start_1'] = np.nan  # w
            data['wave_end_1'] = np.nan  # w
            data['wave_duration_1'] = np.nan  # w
            data['wave_cfr_1'] = np.nan  # w

            country_series = self.data_provider.epidemiology_series \
                [self.data_provider.epidemiology_series['countrycode'] == country].reset_index(drop=True)
            gsi_series = self.data_provider.gsi_table[
                self.data_provider.gsi_table['countrycode'] == country].reset_index(
                drop=True)
            testing_series = self.data_provider.testing[
                self.data_provider.testing['countrycode'] == country].reset_index(
                drop=True)
            # skip country if number of observed days is less than the minimum number of days for a wave
            if len(country_series) < self.config.t_sep_a:
                continue
            # first populate non-wave characteristics
            data['country'] = country_series['country'].iloc[0]
            data['class'], peaks = self._classify(country)
            data['class_coarse'] = 1 if data['class'] <= 2 else (2 if data['class'] <= 4 else 3)
            data['population'] = np.nan if len(
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]) == 0 else \
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]['value'].values[0]
            data['population_density'] = np.nan if len(
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]) == 0 else \
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country][
                    'population_density'].values[0]
            data['gni_per_capita'] = np.nan if len(
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country]) == 0 else \
                self.data_provider.wbi_table[self.data_provider.wbi_table['countrycode'] == country][
                    'gni_per_capita'].values[
                    0]
            data['total_confirmed'] = country_series['confirmed'].iloc[-1]
            data['total_dead'] = country_series['dead'].iloc[-1]
            data['mortality_rate'] = (data['total_dead'] / data['population']) * data['rel_to_constant']
            data['case_rate'] = (data['total_confirmed'] / data['population']) * data['rel_to_constant']
            data['peak_case_rate'] = \
                (country_series['new_per_day_smooth'].max() / data['population']) * data['rel_to_constant']
            data['total_stringency'] = np.nan if len(gsi_series) == 0 else np.trapz(
                y=gsi_series['stringency_index'].dropna(),
                x=[(a - gsi_series['date'].values[0]).days
                   for a in gsi_series['date'][~np.isnan(gsi_series['stringency_index'])]])
            data['t0'] = np.nan if len(
                country_series[country_series['confirmed'] >= self.config.abs_t0_threshold]['date']) == 0 else \
                country_series[country_series['confirmed'] >= self.config.abs_t0_threshold]['date'].iloc[0]
            data['t0_relative'] = np.nan if len(
                country_series[((country_series['confirmed'] /
                                 data[
                                     'population']) * self.config.rel_to_constant >= self.config.rel_t0_threshold)][
                    'date']) == 0 else \
                country_series[((country_series['confirmed'] /
                                 data[
                                     'population']) * self.config.rel_to_constant >= self.config.rel_t0_threshold)][
                    'date'].iloc[
                    0]
            data['t0_1_dead'] = np.nan if len(country_series[country_series['dead'] >= 1]['date']) == 0 else \
                country_series[country_series['dead'] >= 1]['date'].iloc[0]
            data['t0_5_dead'] = np.nan if len(country_series[country_series['dead'] >= 5]['date']) == 0 else \
                country_series[country_series['dead'] >= 5]['date'].iloc[0]
            data['t0_10_dead'] = np.nan if len(country_series[country_series['dead'] >= 10]['date']) == 0 else \
                country_series[country_series['dead'] >= 10]['date'].iloc[0]
            data['testing_available'] = True if len(country_series['new_tests'].dropna()) > 0 else False
            # if t0 not defined all other metrics make no sense
            if pd.isnull(data['t0_10_dead']):
                continue

            '''
            # response time is only defined for the first wave
            if (len(gsi_series) > 0) and (type(peaks) == pd.core.frame.DataFrame) and len(peaks) >= 1:
                sorted_bases = np.sort(
                    np.concatenate((peaks['left_base'].values, peaks['right_base'].values))).astype(int)
                peak_start = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[
                        sorted_bases[np.where(sorted_bases <= int(peaks['location'].iloc[0]))][-1]]
                peak_end = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[
                        sorted_bases[np.where(sorted_bases >= int(peaks['location'].iloc[0]))][0]]
                gsi_first_wave = gsi_series[(gsi_series['date'] <= peak_end) & (gsi_series['date'] >= peak_start)]\
                    .reset_index(drop=True)
                data['stringency_response_time'] = \
                    (gsi_first_wave.iloc[gsi_first_wave['stringency_index'].argmax()]['date'] -  data['t0_1_dead']).days
            elif (len(gsi_series) > 0) and (type(peaks) == pd.core.frame.DataFrame):
                data['stringency_response_time'] = (gsi_series.iloc[gsi_series['stringency_index'].argmax()]['date']
                                                   -  data['t0_1_dead']).days
            else:
                pass
            '''
            if (len(gsi_series) > 0) and (len(gsi_series[gsi_series['c3_cancel_public_events'] == 2]) > 0):
                data['stringency_response_time'] = \
                    (gsi_series[gsi_series['c3_cancel_public_events'] == 2]['date'].iloc[0] - data[
                        't0_10_dead']).days

            if data['testing_available']:
                data['testing_response_time'] = np.nan if \
                    len(testing_series[(testing_series['total_tests'] / data['population']) *
                                       data['rel_to_constant'] >= 10]) == 0 \
                    else \
                    (testing_series[(testing_series['total_tests'] / data['population']) *
                                    data['rel_to_constant'] >= 10]['date'].iloc[0] - data['t0_1_dead']).days
            # for each wave we add characteristics
            if (type(peaks) == pd.core.frame.DataFrame) and len(peaks) > 0:
                sorted_bases = np.sort(
                    np.concatenate((peaks['left_base'].values, peaks['right_base'].values))).astype(
                    int)
                for i, (_, peak) in enumerate(peaks.iterrows(), 1):
                    data['peak_{}'.format(str(i))] = peak['y_position']
                    data['peak_{}_per_rel_to'.format(str(i))] = \
                        (peak['y_position'] / data['population']) * data['rel_to_constant']
                    data['date_peak_{}'.format(str(i))] = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[int(peak['location'])]
                    data['wave_start_{}'.format(str(i))] = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[
                        sorted_bases[np.where(sorted_bases <= int(peak['location']))][-1]]
                    data['wave_end_{}'.format(str(i))] = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[
                        sorted_bases[np.where(sorted_bases >= int(peak['location']))][0]]
                    data['wave_duration_{}'.format(str(i))] = (data['wave_end_{}'.format(str(i))] -
                                                               data['wave_start_{}'.format(str(i))]).days
                    data['wave_cfr_{}'.format(str(i))] = \
                        (country_series[country_series['date'] ==
                                        data['wave_end_{}'.format(str(i))]]['dead'].iloc[0] -
                         country_series[country_series['date'] ==
                                        data['wave_start_{}'.format(str(i))]]['dead'].iloc[0]) / \
                        (country_series[country_series['date'] ==
                                        data['wave_end_{}'.format(str(i))]]['confirmed'].iloc[0] -
                         country_series[country_series['date'] ==
                                        data['wave_start_{}'.format(str(i))]]['confirmed'].iloc[0])
                    continue
            epidemiology_panel = epidemiology_panel.append(data, ignore_index=True)
            continue
        epidemiology_panel.to_csv(os.path.join(self.data_path, 'table_of_results.csv'), index=False)
        return epidemiology_panel

    # pg.mwu abstracts the decision of less than or greater than
    # results after droping NaNs should be the same
    def _mann_whitney(self, data, field='gni_per_capita'):
        x = data[data['class_coarse'] == 1][field].dropna().values
        y = data[~(data['class_coarse'] == 1)][field].dropna().values
        return pg.mwu(x, y, tail='one-sided')

    # waiting implementation 'country', 'countrycode',
    def table_1(self):
        print('Generation Table 1')

        epidemiology_panel = self._get_epi_panel()
        median = epidemiology_panel[
            ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
             'stringency_response_time', 'total_stringency', 'testing_response_time',
             'population_density', 'gni_per_capita']].groupby(by=['class_coarse']).median().T
        quartile_1 = epidemiology_panel[
            ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
             'stringency_response_time', 'total_stringency', 'testing_response_time',
             'population_density', 'gni_per_capita']].groupby(by=['class_coarse']).quantile(0.25).T
        quartile_3 = epidemiology_panel[
            ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
             'stringency_response_time', 'total_stringency', 'testing_response_time',
             'population_density', 'gni_per_capita']].groupby(by=['class_coarse']).quantile(0.75).T
        data = pd.concat(
            [quartile_1, median, quartile_3], keys=['quartile_1', 'median', 'quartile_3'], axis=1).sort_values(
            by=['class_coarse'], axis=1)
        data.to_csv(os.path.join(self.data_path, 'table_1_v1.csv'))
        self._mann_whitney(epidemiology_panel[
                               ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
                                'stringency_response_time', 'total_stringency', 'testing_response_time',
                                'population_density', 'gni_per_capita']].copy(), field='gni_per_capita').to_csv(
            os.path.join(self.data_path, 'mann_whitney_gni.csv'))
        self._mann_whitney(epidemiology_panel[
                               ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
                                'stringency_response_time', 'total_stringency', 'testing_response_time',
                                'population_density', 'gni_per_capita']].copy(),
                           field='stringency_response_time').to_csv(
            os.path.join(self.data_path, 'mann_whitney_si.csv'))
        print('Done')
        return data
