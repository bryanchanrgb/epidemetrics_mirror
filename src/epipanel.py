import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from data_provider import DataProvider
from implementation.config import Config


class EpiPanel:
    def __init__(self, config: Config, data_provider: DataProvider, peaksAndTroughs: dict):
        self.config = config
        self.peaksAndTroughs = peaksAndTroughs
        self.data_provider = data_provider

    def _classify(self, country):
        if country not in self.config.exclude_countries:
            peaksAndTroughs = self.peaksAndTroughs.get(country)
            peak_class = len(self.peaksAndTroughs) + 1 if peaksAndTroughs else 0
            # if the list of peaksAndTroughs is empty we get peak_class = 1 - check if this is accurate
            if peak_class == 1:
                data = self._get_series(country=country, field='new_per_day_smooth')
                if np.nanmax(data['new_per_day_smooth']) < self.class_1_threshold:
                    peak_class = 0
        else:
            return 0, None
        return peak_class, peaksAndTroughs

    # waiting implementation
    def get_epi_panel(self):
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
            data['class'], peaksAndTroughs = self._classify(country)
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
            if (len(gsi_series) > 0) and (type(peaksAndTroughs) == pd.core.frame.DataFrame) and len(peaksAndTroughs) 
            >= 1:
                sorted_bases = np.sort(
                    np.concatenate((peaksAndTroughs['left_base'].values, peaksAndTroughs[
                    'right_base'].values))).astype(int)
                peak_start = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[
                        sorted_bases[np.where(sorted_bases <= int(peaksAndTroughs['location'].iloc[0]))][-1]]
                peak_end = country_series.dropna(
                        subset=['new_per_day_smooth'])['date'].iloc[
                        sorted_bases[np.where(sorted_bases >= int(peaksAndTroughs['location'].iloc[0]))][0]]
                gsi_first_wave = gsi_series[(gsi_series['date'] <= peak_end) & (gsi_series['date'] >= peak_start)]\
                    .reset_index(drop=True)
                data['stringency_response_time'] = \
                    (gsi_first_wave.iloc[gsi_first_wave['stringency_index'].argmax()]['date'] -  data['t0_1_dead']).days
            elif (len(gsi_series) > 0) and (type(peaksAndTroughs) == pd.core.frame.DataFrame):
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
            if (type(peaksAndTroughs) == list) and len(peaksAndTroughs) > 0:
                for peak in peaksAndTroughs:
                    # only run this for peaks
                    if peak['peak_ind'] == 0:
                        continue
                    endOfWaveFound = False
                    # wave number
                    i = int((peak['index'] + 2) / 2)
                    # data relating to the peak
                    data['peak_{}'.format(str(i))] = peak['y_position']
                    data['peak_{}_per_rel_to'.format(str(i))] = \
                        (peak['y_position'] / data['population']) * data['rel_to_constant']
                    data['date_peak_{}'.format(str(i))] = peak['date']
                    # find preceding and following troughs
                    if i == 1:
                        wave_start = np.nan if len(country_series[country_series['confirmed'] >= 1]['date']) == 0 \
                            else \
                            country_series[country_series['confirmed'] >= 1]['date'].iloc[0]
                        data['wave_start_{}'.format(str(i))] = wave_start
                    for trough in peaksAndTroughs:
                        if trough['index'] == peak['index'] - 1:
                            data['wave_start_{}'.format(str(i))] = trough['date']
                        elif trough['index'] == peak['index'] + 1:
                            data['wave_end_{}'.format(str(i))] = trough['date']
                            endOfWaveFound = True
                    if not endOfWaveFound:
                        wave_end = np.nan if len(country_series[country_series['confirmed'] >= 1]['date']) == 0 \
                            else \
                            country_series[country_series['confirmed'] >= 1]['date'].iloc[-1]
                        data['wave_end_{}'.format(str(i))] = wave_end
                    # calculate information relating to the wave
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
        epidemiology_panel.to_csv(os.path.join(self.config.data_path, 'table_of_results.csv'), index=False)
        return epidemiology_panel
