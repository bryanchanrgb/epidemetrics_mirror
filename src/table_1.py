import os

import pandas as pd
import pingouin as pg
from implementation.config import Config


class Table_1:
    def __init__(self, config: Config, epi_panel: pd.core.frame.DataFrame):
        self.config = config
        self.epi_panel = epi_panel

    # pg.mwu abstracts the decision of less than or greater than
    # results after droping NaNs should be the same
    def _mann_whitney(self, data, field='gni_per_capita'):
        x = data[data['class_coarse'] == 1][field].dropna().values
        y = data[~(data['class_coarse'] == 1)][field].dropna().values
        return pg.mwu(x, y, tail='one-sided')

    # waiting implementation 'country', 'countrycode',
    def table_1(self):
        print('Generating Table 1')

        epidemiology_panel = self.epi_panel
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
        data.to_csv(os.path.join(self.config.data_path, 'table_1_v1.csv'))
        self._mann_whitney(epidemiology_panel[
                               ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
                                'stringency_response_time', 'total_stringency', 'testing_response_time',
                                'population_density', 'gni_per_capita']].copy(), field='gni_per_capita').to_csv(
            os.path.join(self.config.data_path, 'mann_whitney_gni.csv'))
        self._mann_whitney(epidemiology_panel[
                               ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
                                'stringency_response_time', 'total_stringency', 'testing_response_time',
                                'population_density', 'gni_per_capita']].copy(),
                           field='stringency_response_time').to_csv(
            os.path.join(self.config.data_path, 'mann_whitney_si.csv'))
        print('Done')
        return data
