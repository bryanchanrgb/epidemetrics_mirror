import numpy as np
import pandas as pd
import datetime
import psycopg2
from tqdm import tqdm
from csaps import csaps
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

class epidemetrics:
    def __init__(self):
        self.source = 'WRD_WHO'
        self.end_date = datetime.date(2021, 4, 1)
        self.ma_window = 7
        self.use_splines = False
        self.smooth = 0.001
        self.flags = {
            'c1_school_closing': 3,
            'c2_workplace_closing': 3,
            'c3_cancel_public_events': 2,
            'c4_restrictions_on_gatherings': 4,
            'c5_close_public_transport': 2,
            'c6_stay_at_home_requirements': 2,
            'c7_restrictions_on_internal_movement': 2,
            'c8_international_travel_controls': 4,
            'h2_testing_policy': 3,
            'h3_contact_tracing': 2
            }
        self.wb_codes = {
            'SP.POP.TOTL': 'value',
            'EN.POP.DNST': 'population_density',
            'NY.GNP.PCAP.PP.KD': 'gni_per_capita',
            'SM.POP.NETM': 'net_migration'
            }
        # convention - rel refers to metrics normalised by population, abs refers to no normalisation
        self.abs_t0_threshold = 1000
        self.abs_prominence_threshold = 55 # minimum prominence
        self.abs_prominence_threshold_dead = 5 # minimum prominence for dead peak detection
        self.rel_t0_threshold = 0.05 # cases per rel_to_constant
        self.rel_prominence_threshold = 0.05 # prominence relative to rel_to_constant
        self.rel_prominence_threshold_dead = 0.0015 # prominence threshold for dead rel_to_constant
        self.rel_prominence_max_threshold = 500 # upper limit on relative prominence
        self.rel_prominence_max_threshold_dead = 50 # upper limit on relative prominencce
        self.rel_to_constant = 10000 # used as population reference for relative t0
        self.prominence_height_threshold = 0.7 # prominence must be above a percentage of the peak height
        self.t_sep_a = 21
        self.v_sep_b = 10 # v separation for sub algorithm B
        self.d_match = 35 # matching window for undetected case waves based on death waves
        self.plot_path = './plots/algorithm_results/'
        self.exclude_countries = ['CMR','COG','GNQ','BWA','ESH'] # countries with low quality data to be ignored
        self.class_1_threshold = 55 # minimum number of absolute cases to be considered going into first wave
        self.class_1_threshold_dead = 5
        self.debug_death_lag = 9 # death lag for case-death ascertainment
        self.debug_countries_of_interest = ['USA', 'GBR', 'BRA', 'IND', 'ESP', 'FRA', 'ZAF']

        '''
        INITIALISE SERVER CONNECTION
        '''
        self.conn = psycopg2.connect(
            host='covid19db.org',
            port=5432,
            dbname='covid19',
            user='covid19',
            password='covid19')
        self.conn.cursor()

        '''
        PULL/PROCESS DATA 
        '''
        self.epidemiology = self._get_epi_table()
        self.testing = self._get_tst_table()
        self.wbi_table = self._get_wbi_table()
        self.epidemiology_series = self._get_epi_series(
            epidemiology=self.epidemiology, testing=self.testing, wbi_table=self.wbi_table)
        self.gsi_table = self._get_gsi_table()
        return

    def _get_epi_table(self):
        '''
        PREPARE EPIDEMIOLOGY TABLE
        '''
        cols = 'countrycode, country, date, confirmed, dead'
        sql_command = 'SELECT ' + cols + \
                      ' FROM epidemiology WHERE adm_area_1 IS NULL AND source = %(source)s AND gid IS NOT NULL'
        epi_table = pd.read_sql(sql_command, self.conn, params={'source':self.source}) \
                            .sort_values(by=['countrycode', 'date'])
        epi_table = epi_table[epi_table['date'] <= self.end_date] \
                            .reset_index(drop=True)
        # checks for any duplication/conflicts in the timeseries
        assert not epi_table[['countrycode', 'date']].duplicated().any()
        epidemiology = pd.DataFrame(columns=[
                            'countrycode', 'country', 'date', 'confirmed', 'new_per_day','dead_per_day'])
        for country in tqdm(epi_table['countrycode'].unique(), desc='Pre-processing Epidemiological Data'):
            data = epi_table[epi_table['countrycode'] == country].set_index('date')
            # cast all dates as datetime date to omit ambiguity
            data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
            # fill gaps in countrycode
            data[['countrycode', 'country']] = data[['countrycode', 'country']].fillna(method='backfill')
            # linearly interpolate gaps in confirmed data
            data['confirmed'] = data['confirmed'].interpolate(method='linear')
            # diff on interpolated data is equivalent to attributing the rise in new cases over two days
            data['new_per_day'] = data['confirmed'].diff(periods=1)
            data.reset_index(inplace=True)
            # for negative values of new cases per day (inaccuracies caused by consolidation) replace with last
            data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index)] = \
                data['new_per_day'].iloc[np.array(data[data['new_per_day'] < 0].index) - 1]
            # fill na with last acceptable value
            data['new_per_day'] = data['new_per_day'].fillna(method='bfill')
            # similarly interpolate death
            data['dead'] = data['dead'].interpolate(method='linear')
            data['dead_per_day'] = data['dead'].diff()
            # data.reset_index(inplace=True)
            data['dead_per_day'].iloc[np.array(data[data['dead_per_day'] < 0].index)] = \
                data['dead_per_day'].iloc[np.array(data[data['dead_per_day'] < 0].index) - 1]
            data['dead_per_day'] = data['dead_per_day'].fillna(method='bfill')
            epidemiology = pd.concat((epidemiology, data)).reset_index(drop=True)
            continue
        return epidemiology

    def _get_epi_series(self, epidemiology, testing, wbi_table):
        # initialise epidemiology time-series data
        epidemiology_series = {
            'countrycode': np.empty(0),
            'country': np.empty(0),
            'date': np.empty(0),
            'confirmed': np.empty(0),
            'new_per_day': np.empty(0),
            'new_per_day_smooth': np.empty(0),
            'dead': np.empty(0),
            'days_since_t0': np.empty(0),
            'new_cases_per_rel_constant': np.empty(0),
            'dead_per_day': np.empty(0),
            'dead_per_day_smooth': np.empty(0),
            'new_deaths_per_rel_constant': np.empty(0),
            'tests': np.empty(0),
            'new_tests': np.empty(0),
            'new_tests_smooth': np.empty(0),
            'positive_rate': np.empty(0),
            'positive_rate_smooth': np.empty(0),
            'days_since_t0_pop': np.empty(0),
            'days_since_t0_1_dead': np.empty(0),
            'days_since_t0_5_dead': np.empty(0),
            'days_since_t0_10_dead': np.empty(0),
            'case_death_ascertainment': np.empty(0)
        }

        for country in tqdm(np.sort(epidemiology['countrycode'].unique()),
                            desc='Processing Epidemiological Time Series Data'):
            epi_data = epidemiology[epidemiology['countrycode'] == country]
            tst_data = testing[testing['countrycode'] == country]
            # we want a master spreadsheet
            tests = np.repeat(np.nan, len(epi_data))
            new_tests = np.repeat(np.nan, len(epi_data))
            new_tests_smooth = np.repeat(np.nan, len(epi_data))
            positive_rate = np.repeat(np.nan, len(epi_data))
            positive_rate_smooth = np.repeat(np.nan, len(epi_data))
            # if we want to run our analysis through a 7d moving average or a spline fit
            if self.use_splines:
                x = np.arange(len(epi_data['date']))
                y = epi_data['new_per_day'].values
                ys = csaps(x, y, x, smooth=self.smooth)
                z = epi_data['dead_per_day'].values
                zs = csaps(x, z, x, smooth=self.smooth)
            else:
                ys = epi_data[['new_per_day', 'date']].rolling(window=self.ma_window, on='date').mean()['new_per_day']
                zs = epi_data[['dead_per_day', 'date']].rolling(window=self.ma_window, on='date').mean()['dead_per_day']
            # preparing testing data based metrics
            if len(tst_data) > 1:
                tests = epi_data[['date']].merge(
                    tst_data[['date', 'total_tests']], how='left', on='date')['total_tests'].values
                # if testing data has new_tests_smoothed, use this
                if sum(~pd.isnull(tst_data['new_tests_smoothed'])) > 0:
                    new_tests_smooth = epi_data[['date']].merge(
                        tst_data[['date', 'new_tests_smoothed']], how='left', on='date')['new_tests_smoothed'].values
                if sum(~pd.isnull(tst_data['new_tests'])) > 0:
                    new_tests = epi_data[['date']].merge(
                        tst_data[['date', 'new_tests']], how='left', on='date')['new_tests'].values
                else:
                    new_tests = new_tests_smooth

                if sum(~pd.isnull(tst_data['new_tests_smoothed'])) == 0 and sum(~pd.isnull(tst_data['new_tests'])) > 0:
                    # if there is no data in new_tests_smoothed, compute 7 day moving average
                    new_tests_smooth = epi_data[['date']] \
                    .merge(tst_data[['date', 'new_tests']], how='left', on='date')[['new_tests', 'date']] \
                    .rolling(window=7, on='date').mean()['new_tests']
                positive_rate[~np.isnan(new_tests)] = epi_data['new_per_day'][~np.isnan(new_tests)] / new_tests[
                    ~np.isnan(new_tests)]
                positive_rate[positive_rate > 1] = np.nan
                positive_rate_smooth = np.array(pd.Series(positive_rate).rolling(window=7).mean())
            # accessing population data from wbi_table
            population = np.nan if len(wbi_table[wbi_table['countrycode'] == country]['value']) == 0 else \
                wbi_table[wbi_table['countrycode'] == country]['value'].iloc[0]
            # two definitions of t0 use where appropriate
            # t0 absolute ~= 1000 total cases or t0 relative = 0.05 per rel_to_constant
            t0 = np.nan if len(epi_data[epi_data['confirmed'] >= self.abs_t0_threshold]['date']) == 0 else \
                epi_data[epi_data['confirmed'] >= self.abs_t0_threshold]['date'].iloc[0]
            t0_relative = np.nan if len(
                epi_data[((epi_data['confirmed'] / population) * self.rel_to_constant) >= self.rel_t0_threshold]) == 0 else \
                epi_data[((epi_data['confirmed'] / population) * self.rel_to_constant) >= self.rel_t0_threshold]['date'].iloc[0]
            # t0_k_dead represents day first k total dead was reported
            t0_1_dead = np.nan if len(epi_data[epi_data['dead'] >= 1]['date']) == 0 else \
                epi_data[epi_data['dead'] >= 1]['date'].iloc[0]
            t0_5_dead = np.nan if len(epi_data[epi_data['dead'] >= 5]['date']) == 0 else \
                epi_data[epi_data['dead'] >= 5]['date'].iloc[0]
            t0_10_dead = np.nan if len(epi_data[epi_data['dead'] >= 10]['date']) == 0 else \
                epi_data[epi_data['dead'] >= 10]['date'].iloc[0]
            # index days since absolute t0, relative t0, k deaths
            days_since_t0 = np.repeat(np.nan, len(epi_data)) if pd.isnull(t0) else \
                np.array([(date - t0).days for date in epi_data['date'].values])
            days_since_t0_relative = np.repeat(np.nan, len(epi_data)) if pd.isnull(t0_relative) else \
                np.array([(date - t0_relative).days for date in epi_data['date'].values])
            days_since_t0_1_dead = np.repeat(np.nan, len(epi_data)) if pd.isnull(t0_1_dead) else \
                np.array([(date - t0_1_dead).days for date in epi_data['date'].values])
            days_since_t0_5_dead = np.repeat(np.nan, len(epi_data)) if pd.isnull(t0_5_dead) else \
                np.array([(date - t0_5_dead).days for date in epi_data['date'].values])
            days_since_t0_10_dead = np.repeat(np.nan, len(epi_data)) if pd.isnull(t0_10_dead) else \
                np.array([(date - t0_10_dead).days for date in epi_data['date'].values])
            # again rel constant represents a population threhsold - 10,000 in the default case
            new_cases_per_rel_constant = self.rel_to_constant * (ys / population)
            new_deaths_per_rel_constant = self.rel_to_constant * (zs / population)
            # compute case-death ascertaintment
            case_death_ascertainment = (epi_data['confirmed'].astype(int) /
                                        epi_data['dead'].astype(int).shift(-9).replace(0, np.nan)).values
            # upsert processed data
            epidemiology_series['countrycode'] = np.concatenate((
                epidemiology_series['countrycode'], epi_data['countrycode'].values))
            epidemiology_series['country'] = np.concatenate(
                (epidemiology_series['country'], epi_data['country'].values))
            epidemiology_series['date'] = np.concatenate(
                (epidemiology_series['date'], epi_data['date'].values))
            epidemiology_series['confirmed'] = np.concatenate(
                (epidemiology_series['confirmed'], epi_data['confirmed'].values))
            epidemiology_series['new_per_day'] = np.concatenate(
                (epidemiology_series['new_per_day'], epi_data['new_per_day'].values))
            epidemiology_series['new_per_day_smooth'] = np.concatenate(
                (epidemiology_series['new_per_day_smooth'], ys))
            epidemiology_series['dead'] = np.concatenate(
                (epidemiology_series['dead'], epi_data['dead'].values))
            epidemiology_series['dead_per_day'] = np.concatenate(
                (epidemiology_series['dead_per_day'], epi_data['dead_per_day'].values))
            epidemiology_series['dead_per_day_smooth'] = np.concatenate(
                (epidemiology_series['dead_per_day_smooth'], zs))
            epidemiology_series['days_since_t0'] = np.concatenate(
                (epidemiology_series['days_since_t0'], days_since_t0))
            epidemiology_series['new_cases_per_rel_constant'] = np.concatenate(
                (epidemiology_series['new_cases_per_rel_constant'], new_cases_per_rel_constant))
            epidemiology_series['new_deaths_per_rel_constant'] = np.concatenate(
                (epidemiology_series['new_deaths_per_rel_constant'], new_deaths_per_rel_constant))
            epidemiology_series['tests'] = np.concatenate(
                (epidemiology_series['tests'], tests))
            epidemiology_series['new_tests'] = np.concatenate(
                (epidemiology_series['new_tests'], new_tests))
            epidemiology_series['new_tests_smooth'] = np.concatenate(
                (epidemiology_series['new_tests_smooth'], new_tests_smooth))
            epidemiology_series['positive_rate'] = np.concatenate(
                (epidemiology_series['positive_rate'], positive_rate))
            epidemiology_series['positive_rate_smooth'] = np.concatenate(
                (epidemiology_series['positive_rate_smooth'], positive_rate_smooth))
            epidemiology_series['days_since_t0_pop'] = np.concatenate(
                (epidemiology_series['days_since_t0_pop'], days_since_t0_relative))
            epidemiology_series['days_since_t0_1_dead'] = np.concatenate(
                (epidemiology_series['days_since_t0_1_dead'], days_since_t0_1_dead))
            epidemiology_series['days_since_t0_5_dead'] = np.concatenate(
                (epidemiology_series['days_since_t0_5_dead'], days_since_t0_5_dead))
            epidemiology_series['days_since_t0_10_dead'] = np.concatenate(
                (epidemiology_series['days_since_t0_10_dead'], days_since_t0_10_dead))
            epidemiology_series['case_death_ascertainment'] = np.concatenate(
                (epidemiology_series['case_death_ascertainment'], case_death_ascertainment))
            continue

        return pd.DataFrame.from_dict(epidemiology_series)
    # waiting implementation
    def _get_epi_panel(self):
        '''epidemiology_static = pd.DataFrame(
            columns=['countrycode', 'country', 'class', 'population',
                     't0', 't0_relative', 't0_1_dead','t0_5_dead',
                     't0_10_dead', 'peak_1', 'peak_2', 'date_peak_1',
                     'date_peak_2', 'first_wave_start', 'first_wave_end', 'duration_first_wave',
                     'second_wave_start', 'second_wave_end','last_confirmed', 'last_dead',
                     'testing_available','peak_1_cfr','peak_2_cfr', 'dead_class','tests_class'])'''
        epidemiology_panel = pd.DataFrame()
        # wave parameters marked a w
        for country in tqdm(np.sort(self.epidemiology_series['countrycode'].unique()),
                            desc='Preparing Epidemiological Results Table'):
            data = dict()
            data['countrycode'] = country
            data['country'] = np.nan
            data['class'] = np.nan
            data['class_coarse'] = np.nan # one, two, three or more waves
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
            data['testing_response_time'] = np.nan # days to reach 10 tests per rel_to
            data['t0'] = np.nan
            data['t0_relative'] = np.nan
            data['t0_1_dead'] = np.nan
            data['t0_5_dead'] = np.nan
            data['t0_10_dead'] = np.nan
            data['testing_available'] = np.nan
            data['rel_to_constant'] = self.rel_to_constant
            data['peak_1'] = np.nan # w
            data['peak_1_per_rel_to'] = np.nan # w
            data['date_peak_1'] = np.nan # w
            data['wave_start_1'] = np.nan # w
            data['wave_end_1'] = np.nan # w
            data['wave_duration_1'] = np.nan # w
            data['wave_cfr_1'] = np.nan # w

            country_series = self.epidemiology_series \
            [self.epidemiology_series['countrycode'] == country].reset_index(drop=True)
            gsi_series = self.gsi_table[self.gsi_table['countrycode'] == country].reset_index(drop=True)
            testing_series = self.testing[self.testing['countrycode'] == country].reset_index(drop=True)
            # skip country if number of observed days is less than the minimum number of days for a wave
            if len(country_series) < self.t_sep_a:
                continue
            # first populate non-wave characteristics
            data['country'] = country_series['country'].iloc[0]
            data['class'], peaks = self._classify(country)
            data['class_coarse'] = 1 if data['class'] <= 2 else (2 if data['class'] <= 4 else 3)
            data['population'] = np.nan if len(self.wbi_table[self.wbi_table['countrycode'] == country]) == 0 else \
                self.wbi_table[self.wbi_table['countrycode'] == country]['value'].values[0]
            data['population_density'] = np.nan if len(self.wbi_table[self.wbi_table['countrycode'] == country]) == 0 else \
                self.wbi_table[self.wbi_table['countrycode'] == country]['population_density'].values[0]
            data['gni_per_capita'] = np.nan if len(self.wbi_table[self.wbi_table['countrycode'] == country]) == 0 else \
                self.wbi_table[self.wbi_table['countrycode'] == country]['gni_per_capita'].values[0]
            data['total_confirmed'] = country_series['confirmed'].iloc[-1]
            data['total_dead'] = country_series['dead'].iloc[-1]
            data['mortality_rate'] = (data['total_dead'] / data['population']) * data['rel_to_constant']
            data['case_rate'] = (data['total_confirmed'] / data['population']) * data['rel_to_constant']
            data['peak_case_rate'] = \
                (country_series['new_per_day_smooth'].max() / data['population']) * data['rel_to_constant']
            data['total_stringency'] = np.nan if len(gsi_series) == 0 else np.trapz(
                y=gsi_series['stringency_index'].dropna(),
                x=[(a-gsi_series['date'].values[0]).days
                   for a in gsi_series['date'][~np.isnan(gsi_series['stringency_index'])]])
            data['t0'] = np.nan if len(
                country_series[country_series['confirmed'] >= self.abs_t0_threshold]['date']) == 0 else \
                country_series[country_series['confirmed'] >= self.abs_t0_threshold]['date'].iloc[0]
            data['t0_relative'] = np.nan if len(
                country_series[((country_series['confirmed'] /
                                data['population']) * self.rel_to_constant >= self.rel_t0_threshold)]['date']) == 0 else \
                country_series[((country_series['confirmed'] /
                                data['population']) * self.rel_to_constant >= self.rel_t0_threshold)]['date'].iloc[0]
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
                    (gsi_series[gsi_series['c3_cancel_public_events'] == 2]['date'].iloc[0] - data['t0_10_dead']).days

            if data['testing_available']:
                data['testing_response_time'] = np.nan if \
                    len(testing_series[(testing_series['total_tests'] / data['population']) *
                                       data['rel_to_constant'] >= 10]) == 0 \
                    else \
                    (testing_series[(testing_series['total_tests'] / data['population']) *
                                   data['rel_to_constant'] >= 10]['date'].iloc[0] - data['t0_1_dead']).days
            # for each wave we add characteristics
            if (type(peaks) == pd.core.frame.DataFrame) and len(peaks) > 0:
                sorted_bases = np.sort(np.concatenate((peaks['left_base'].values, peaks['right_base'].values))).astype(int)
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
        epidemiology_panel.to_csv('./table_of_results.csv',index=False)
        return epidemiology_panel

    def _get_gsi_table(self):
        '''
        PREPARE GOVERNMENT RESPONSE TABLE
        '''
        cols = 'countrycode, country, date, stringency_index, ' + ', '.join(list(self.flags.keys()))
        sql_command = """SELECT """ + cols + """ FROM government_response"""
        gsi_table = pd.read_sql(sql_command, self.conn) \
                            .sort_values(by=['countrycode', 'date']) \
                            .filter(items=['countrycode', 'country', 'date', 'stringency_index'] +
                                          list(self.flags.keys())) \
                            .reset_index(drop=True)
        gsi_table = gsi_table.drop_duplicates(subset=['countrycode', 'date'])
        assert not gsi_table[['countrycode', 'date']].duplicated().any()
        return gsi_table

    def _get_wbi_table(self):
        '''
        PREPARE WORLD BANK STATISTICS
        '''
        sql_command = 'SELECT countrycode, indicator_code, value FROM world_bank WHERE ' \
                      'adm_area_1 IS NULL AND indicator_code IN %(indicator_code)s'
        raw_wbi_table = pd.read_sql(sql_command, self.conn,
                                params={'indicator_code': tuple(self.wb_codes.keys())}).dropna()
        raw_wbi_table = raw_wbi_table.sort_values(by=['countrycode'], ascending=[True]).reset_index(drop=True)
        assert not raw_wbi_table[['countrycode', 'indicator_code']].duplicated().any()

        wbi_table = pd.DataFrame()
        for country in raw_wbi_table['countrycode'].unique():
            data = dict()
            data['countrycode'] = country
            for indicator in self.wb_codes.keys():
                if len(raw_wbi_table[(raw_wbi_table['countrycode'] == country) &
                                     (raw_wbi_table['indicator_code'] == indicator)]['value']) == 0:
                    continue
                data[self.wb_codes[indicator]] = raw_wbi_table[
                    (raw_wbi_table['countrycode'] == country) &
                    (raw_wbi_table['indicator_code'] == indicator)]['value'].iloc[0]
            wbi_table = wbi_table.append(data, ignore_index=True)
        wbi_table['net_migration'] = wbi_table['net_migration'].abs()
        return wbi_table

    def _get_tst_table(self):
        '''
        PREPARE TESTING TABLE
        '''
        owid_source = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
        tst_table = pd.read_csv(owid_source, parse_dates=['date'])[
                        ['iso_code', 'date', 'total_tests', 'new_tests', 'new_tests_smoothed', 'positive_rate']] \
                        .rename(columns={'iso_code': 'countrycode'})
        tst_table['date'] = tst_table['date'].apply(lambda x: x.date())
        tst_table = tst_table[tst_table['date'] <= self.end_date].reset_index(drop=True)
        # filters out some odd columns in the countrycode column
        countries = [country for country in tst_table['countrycode'].unique()
                     if not (pd.isnull(country)) and (len(country) == 3)]
        # initialise results dataframe
        testing = pd.DataFrame(
            columns=['countrycode', 'date', 'total_tests', 'new_tests', 'new_tests_smoothed', 'positive_rate'])
        for country in tqdm(countries, desc='Pre-processing Testing Data'):
            data = tst_table[tst_table['countrycode'] == country].reset_index(drop=True)
            # filter out countries without any testing data
            if len(data['new_tests'].dropna()) == 0:
                continue
            # slice all testing data from when the first and last available date
            data = data.iloc[
                   data[(data['new_tests'].notnull()) | (data['new_tests_smoothed'].notnull())].index[0]:
                   data[(data['new_tests'].notnull()) | (data['new_tests_smoothed'].notnull())].index[-1]].set_index(
                'date')
            if len(data) > 0:
                # reindexing to include all dates available
                data = data.reindex([x.date() for x in pd.date_range(data.index.values[0], data.index.values[-1])])
                # nans from reindexing filled in using linear interpolation
                data[['countrycode']] = data[['countrycode']].fillna(method='backfill')
                data[['total_tests']] = data[['total_tests']].interpolate(method='linear')
                data[['new_tests']] = data[['new_tests']].interpolate(method='linear')
                data.reset_index(inplace=True)
                testing = pd.concat((testing, data), ignore_index=True)
        return testing

    def _sub_algorithm_a(self, country, field='new_per_day_smooth', plot=False, override=None):
        if type(override) == pd.core.frame.DataFrame:
            data = override.copy()
        else:
            data = self._get_series(country, field)
        peak = find_peaks(data[field].values, prominence=0, distance=1)
        trough = find_peaks([-x for x in data[field].values], prominence=0, distance=1)
        sub_a = pd.DataFrame(data=np.transpose([np.append(data.index[peak[0]], data.index[trough[0]]),
                                      np.append(peak[1]['prominences'], trough[1]['prominences']),
                                      np.append(data.index[peak[1]['left_bases']], data.index[trough[1]['left_bases']]),
                                      np.append(data.index[peak[1]['right_bases']], data.index[trough[1]['right_bases']])]),
                             columns=['location', 'prominence','left_base','right_base'])
        sub_a['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
        sub_a = sub_a.sort_values(by='location').reset_index(drop=True)
        # if there are fewer than 3 points, the algorithm cannot be run, return nothing
        if len(sub_a) < 3:
            results = pd.DataFrame(columns=['location', 'prominence', 'duration','left_base','right_base', 'index', 'peak_ind'])
        else:
            # calculate the duration of extrema i as the distance between the extrema to the left and to the right of i
            for i in range(1, len(sub_a) - 1):
                sub_a.loc[i, 'duration'] = sub_a.loc[i + 1, 'location'] - sub_a.loc[i - 1, 'location']
            # remove peaks and troughs until the smallest duration meets T_SEP
            while np.nanmin(sub_a['duration']) < self.t_sep_a and len(sub_a) >= 3:
                # sort the peak/trough candidates by prominence, retaining the location index
                sub_a = sub_a.sort_values(by=['prominence', 'duration']).reset_index(drop=False)
                # remove the lowest prominence candidate with duration < T_SEP
                x = min(sub_a[sub_a['duration'] < self.t_sep_a].index)
                i = sub_a.loc[x, 'index']
                sub_a.drop(index=x, inplace=True)
                # remove whichever adjacent candidate has the lower prominence. If tied, remove the earlier.
                if sub_a.loc[sub_a['index'] == i + 1, 'prominence'].values[0] >= \
                        sub_a.loc[sub_a['index'] == i - 1, 'prominence'].values[0]:
                    sub_a = sub_a.loc[sub_a['index'] != i - 1, ['prominence', 'location', 'peak_ind', 'left_base', 'right_base']]
                else:
                    sub_a = sub_a.loc[sub_a['index'] != i + 1, ['prominence', 'location', 'peak_ind', 'left_base', 'right_base']]
                # re-sort by location and recalculate duration
                sub_a = sub_a.sort_values(by='location').reset_index(drop=True)
                if len(sub_a) >= 3:
                    for i in range(1, len(sub_a) - 1):
                        sub_a.loc[i, 'duration'] = sub_a.loc[i + 1, 'location'] - sub_a.loc[i - 1, 'location']
                else:
                    sub_a['duration'] = np.nan
            results = sub_a.copy()

        if plot:
            plt.plot(data[field].values)
            plt.scatter(sub_a['location'].values,
                        data[field].values[sub_a['location'].values.astype(int)], color='red', marker='o')
        # results returns a set of peaks and troughs which are at least a minimum distance apart
        return results

    def _sub_algorithm_b(self, sub_a, country, field='new_per_day_smooth', plot=False):
        data = self._get_series(country, field)
        sub_b_flag = True
        # avoid overwriting sub_a when values replaced by t0 and t1
        results = sub_a.copy()
        # dictionary to hold boundaries for peak-trough pairs too close to each other
        og_dict = dict()
        while sub_b_flag == True:
            # separation here refers to temporal distance S_i
            results.loc[0:len(results) - 2, 'separation'] = np.diff(results['location'])
            results.loc[:, 'y_position'] = data[field][results['location']].values
            # compute vertical distance V_i
            results.loc[0:len(results) - 2, 'y_distance'] = [abs(x) for x in np.diff(results['y_position'])]
            # sort in ascending order of height
            results = results.sort_values(by='y_distance').reset_index(drop=False)
            # set to false until we find an instance where S_i < t_sep_a / 2 and V_i > v_sep_b
            sub_b_flag = False
            for x in results.index:
                if results.loc[x, 'y_distance'] >= self.v_sep_b and results.loc[x, 'separation'] < self.t_sep_a / 2:
                    sub_b_flag = True
                    i = results.loc[x, 'index']
                    og_0 = results.loc[results['index'] == i, 'location'].values[0]
                    og_1 = results.loc[results['index'] == i + 1, 'location'].values[0]
                    # creating boundaries t_0 and t_1 around the peak-trough pair
                    t_0 = np.floor((og_0 + og_1 - self.t_sep_a) / 2)
                    t_1 = np.floor((og_0 + og_1 + self.t_sep_a) / 2)
                    # store the original locations to restore them at the end
                    og_dict[len(og_dict)] = [og_0, t_0, og_1, t_1]
                    # setting the peak locations to the boundaries to be filtered by sub_algorithm_a
                    results.loc[results['index'] == i, 'location'] = t_0
                    results.loc[results['index'] == i + 1, 'location'] = t_1
                    # run the indices list (adding start and end of the time series to the list) through find_peaks again
                    locations = np.clip(np.sort(np.append(results['location'],
                                                  [min(data[field][~np.isnan(data[field].values)].index),
                                                   max(data[field][~np.isnan(data[field])].index)])), 0, len(data) - 1)
                    # run the resulting peaks through sub algorithm A again
                    results = self._sub_algorithm_a(country=country, field=field,
                                                    plot=False, override=data.iloc[locations])
                    break

        for g in sorted(og_dict, reverse=True):
            results.loc[results['location'] == og_dict[g][1], 'location'] = og_dict[g][0]
            results.loc[results['location'] == og_dict[g][3], 'location'] = og_dict[g][2]

        if plot:
            fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex='all')
            # plot peaks-trough pairs from sub_a
            ax0.set_title('After Sub Algorithm A')
            ax0.plot(data[field].values)
            ax0.scatter(sub_a['location'].values,
                        data[field].values[sub_a['location'].values.astype(int)], color='red', marker='o')
            # plot peaks-trough pairs from sub_b
            ax1.set_title('After Sub Algorithm B')
            ax1.plot(data[field].values)
            ax1.scatter(results['location'].values,
                        data[field].values[results['location'].values.astype(int)], color='red', marker='o')
        return results

    def _sub_algorithm_c(self, sub_a, sub_b, country, field='new_per_day_smooth', plot=False):
        data = self._get_series(country, field)
        population = self.wbi_table[self.wbi_table['countrycode'] == country]['value'].values[0]
        if field == 'dead_per_day_smooth':
            abs_prominence_threshold = self.abs_prominence_threshold_dead
            rel_prominence_threshold = self.rel_prominence_threshold_dead
            rel_prominence_max_threshold = self.rel_prominence_max_threshold_dead
        else:
            abs_prominence_threshold = self.abs_prominence_threshold
            rel_prominence_threshold = self.rel_prominence_threshold
            rel_prominence_max_threshold = self.rel_prominence_max_threshold

        # prominence filter will use the larger of the absolute prominence threshold and relative prominence threshold
        # we cap the relative prominence threshold to rel_prominence_max_threshold
        prominence_threshold = max(abs_prominence_threshold,
                                   min(rel_prominence_threshold * population / self.rel_to_constant,
                                       rel_prominence_max_threshold))
        results = sub_b.copy()
        results = results.sort_values(by='location').reset_index(drop=True)
        # filter out troughs and peaks below prominence threshold
        results = results[(results['peak_ind'] == 1) & (results['prominence'] >= prominence_threshold)]
        # filter out relatively low prominent peaks
        results = results[(results['prominence'] >= self.prominence_height_threshold * results['y_position'])]

        if plot:
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
        return results

    def _sub_algorithm_e(self, cases_sub_a, cases_sub_b, cases_sub_c, deaths_sub_c, country, plot=False):
        # basically look into the output of sub algorithm b for cases and sub algorithm c for deaths
        # if there is a case peak in self.d_match days before ignore
        # else use the most prominent peak in the d days before
        # if the first death peak is before d_match then no use of doing this
        if deaths_sub_c['location'].min() < self.d_match:
            return cases_sub_c

        data = self._get_series(country, field='new_per_day_smooth')
        results = cases_sub_c.copy()
        for i, death_peak in enumerate(deaths_sub_c['location']):
            # check if for this death peak there is a peak in cases between (death_peak - d_match) and death_peak
            # if peak already there continue
            if np.any([True if (x >= death_peak - self.d_match) and (x <= death_peak)
                       else False for x in cases_sub_c['location']]):
                continue
            # if peak in cases_sub_b output use the most prominent one
            elif np.any([True if (x >= death_peak - self.d_match) and (x <= death_peak)
                        else False for x in cases_sub_b['location']]):
                # potential candidates for peaks are those within range in cases_sub_b
                candidates = cases_sub_b[
                    (cases_sub_b['location'] >= death_peak - self.d_match) & (cases_sub_b['location'] <= death_peak)]
                results = results.append(candidates.loc[candidates.idxmax()['prominence']])
                continue
            # if nothing, could use max - but might violate t_sep rule...
            else:
                continue
        results = results.sort_values(by=['location'])

        if plot:
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
        return results

    def _find_peaks(self, country, plot=False, save=False):
        # match parameter tries to use death waves to detect case waves under sub_algorithm_e
        cases = self._get_series(country=country, field='new_per_day_smooth')
        if len(cases) == 0:
            raise ValueError
        cases_sub_a = self._sub_algorithm_a(country=country, field='new_per_day_smooth')
        cases_sub_b = self._sub_algorithm_b(cases_sub_a, country=country, field='new_per_day_smooth')
        cases_sub_c = self._sub_algorithm_c(
            sub_a=cases_sub_a, sub_b=cases_sub_b, country=country, field='new_per_day_smooth')
        # compute equivalent series for deaths
        deaths = self._get_series(country=country, field='dead_per_day_smooth')
        if len(deaths) == 0:
            raise ValueError
        deaths_sub_a = self._sub_algorithm_a(country=country, field='dead_per_day_smooth')
        deaths_sub_b = self._sub_algorithm_b(deaths_sub_a, country=country, field='dead_per_day_smooth')
        deaths_sub_c = self._sub_algorithm_c(
            sub_a=deaths_sub_a, sub_b=deaths_sub_b, country=country, field='dead_per_day_smooth')
        # run sub algorithm e
        cases_sub_e = self._sub_algorithm_e(cases_sub_a, cases_sub_b, cases_sub_c, deaths_sub_c, country=country)
        # compute plots
        if plot:
            fig, axs = plt.subplots(nrows=2, ncols=4, sharex=True, figsize=(14, 7))
            plt.suptitle(country)

            axs[0, 0].set_title('Cases After Sub Algorithm A')
            axs[0, 0].plot(cases['new_per_day_smooth'].values)
            axs[0, 0].scatter(cases_sub_a['location'].values,
                              cases['new_per_day_smooth'].values[
                                  cases_sub_a['location'].values.astype(int)], color='red', marker='o')
            axs[0, 0].get_xaxis().set_visible(False)
            axs[0, 0].get_yaxis().set_visible(False)

            axs[0, 1].set_title('Cases After Sub Algorithm B')
            axs[0, 1].plot(cases['new_per_day_smooth'].values)
            axs[0, 1].scatter(cases_sub_b['location'].values,
                              cases['new_per_day_smooth'].values[
                                  cases_sub_b['location'].values.astype(int)], color='red', marker='o')
            axs[0, 1].get_xaxis().set_visible(False)
            axs[0, 1].get_yaxis().set_visible(False)

            axs[0, 2].set_title('Cases After Sub Algorithm C&D')
            axs[0, 2].plot(cases['new_per_day_smooth'].values)
            axs[0, 2].scatter(cases_sub_c['location'].values,
                              cases['new_per_day_smooth'].values[
                                  cases_sub_c['location'].values.astype(int)], color='red', marker='o')
            axs[0, 2].get_xaxis().set_visible(False)
            axs[0, 2].get_yaxis().set_visible(False)

            axs[0, 3].set_title('Cases After Sub Algorithm E')
            axs[0, 3].plot(cases['new_per_day_smooth'].values)
            axs[0, 3].scatter(cases_sub_e['location'].values,
                              cases['new_per_day_smooth'].values[
                                  cases_sub_e['location'].values.astype(int)], color='red', marker='o')
            axs[0, 3].get_xaxis().set_visible(False)
            axs[0, 3].get_yaxis().set_visible(False)

            axs[1, 0].set_title('Deaths After Sub Algorithm A')
            axs[1, 0].plot(deaths['dead_per_day_smooth'].values)
            axs[1, 0].scatter(deaths_sub_a['location'].values,
                              deaths['dead_per_day_smooth'].values[
                                  deaths_sub_a['location'].values.astype(int)], color='red', marker='o')
            axs[1, 0].get_xaxis().set_visible(False)
            axs[1, 0].get_yaxis().set_visible(False)

            axs[1, 1].set_title('Deaths After Sub Algorithm B')
            axs[1, 1].plot(deaths['dead_per_day_smooth'].values)
            axs[1, 1].scatter(deaths_sub_b['location'].values,
                              deaths['dead_per_day_smooth'].values[
                                  deaths_sub_b['location'].values.astype(int)], color='red', marker='o')
            axs[1, 1].get_xaxis().set_visible(False)
            axs[1, 1].get_yaxis().set_visible(False)

            axs[1, 2].set_title('Deaths After Sub Algorithm C&D')
            axs[1, 2].plot(deaths['dead_per_day_smooth'].values)
            axs[1, 2].scatter(deaths_sub_c['location'].values,
                              deaths['dead_per_day_smooth'].values[
                                  deaths_sub_c['location'].values.astype(int)], color='red', marker='o')
            axs[1, 2].get_xaxis().set_visible(False)
            axs[1, 2].get_yaxis().set_visible(False)

            axs[1, 3].get_xaxis().set_visible(False)
            axs[1, 3].get_yaxis().set_visible(False)
            fig.tight_layout()

            if save:
                plt.savefig(self.plot_path + country + '.png')
                plt.close('all')
        return cases_sub_e
    # check with ZWE, ?
    def _classify(self, country, field='new_per_day_smooth'):
        data = self._get_series(country=country, field=field)
        # class 0 reserved for misbehaving cases
        if (len(data) < 3) or \
                (country in self.exclude_countries) or \
                not(country in self.wbi_table['countrycode'].values):
            return 0, None
        # method _find_peaks is only supported for new_cases_per_day as the cross-validation step requires death per day
        # for alternative fields the output of sub_algorithm_c is used
        if field == 'new_per_day_smooth':
            deaths = self._get_series(country=country, field='dead_per_day_smooth')
            if len(deaths) < 3:
                return 0, None
            genuine_peaks = self._find_peaks(country, plot=False, save=False)
        else:
            sub_a = self._sub_algorithm_a(country, field=field, plot=False)
            sub_b = self._sub_algorithm_b(sub_a, country, field=field, plot=False)
            genuine_peaks = self._sub_algorithm_c(sub_a, sub_b, country, field=field, plot=False)

        population = self.wbi_table[self.wbi_table['countrycode'] == country]['value'].values[0]
        if field == 'dead_per_day_smooth':
            abs_prominence_threshold = self.abs_prominence_threshold_dead
            rel_prominence_threshold = self.rel_prominence_threshold_dead
            rel_prominence_max_threshold = self.rel_prominence_max_threshold_dead
            class_1_threshold = self.class_1_threshold_dead
        else:
            abs_prominence_threshold = self.abs_prominence_threshold
            rel_prominence_threshold = self.rel_prominence_threshold
            rel_prominence_max_threshold = self.rel_prominence_max_threshold
            class_1_threshold = self.class_1_threshold

        # prominence filter will use the larger of the absolute prominence threshold and relative prominence threshold
        # we cap the relative prominence threshold to rel_prominence_max_threshold
        prominence_threshold = max(abs_prominence_threshold,
                                   min(rel_prominence_threshold * population / self.rel_to_constant,
                                       rel_prominence_max_threshold))

        peak_class = 2 * len(genuine_peaks)
        # if the last value is able to meet the constraints from sub algorithm C, we can
        if (peak_class > 0) and (genuine_peaks['location'].iloc[-1] < len(data)):
            last_peak_date = data['date'].values[int(genuine_peaks['location'].iloc[-1])]
            trough_value = min(data.loc[data['date'] > last_peak_date, 'new_per_day_smooth'])
            trough_date = data[data['date'] > last_peak_date]['date'].iloc[
                int(np.argmin(data.loc[data['date'] > last_peak_date, 'new_per_day_smooth']))]
            max_after_trough = np.nanmax(
                data.loc[data['date'] >= trough_date, 'new_per_day_smooth'])
            if (max_after_trough - trough_value >= prominence_threshold) \
                    and (max_after_trough - trough_value >= self.prominence_height_threshold * max_after_trough):
                peak_class += 1
        elif (peak_class == 0) and not(country in self.exclude_countries):
            if np.nanmax(data['new_per_day_smooth']) >= class_1_threshold:
                peak_class += 1
        else:
            pass
        return peak_class, genuine_peaks

    def main(self):
        countries = self.testing['countrycode'].unique()
        for country in tqdm(countries, desc='Plotting all charts'):
            try:
                self._find_peaks(country, plot=True, save=True)
            except:
                print(country)
        return
    # waiting implementation 'country', 'countrycode',
    def table_1(self):
        epidemiology_panel = self._get_epi_panel()
        data = epidemiology_panel[
            ['class_coarse', 'mortality_rate', 'case_rate', 'peak_case_rate',
             'stringency_response_time', 'total_stringency', 'testing_response_time',
             'population_density', 'gni_per_capita']].groupby(by=['class_coarse']).mean().T
        data.to_csv('./data/table_1.csv')
        return data

    def _get_series(self, country, field):
        return self.epidemiology_series[self.epidemiology_series['countrycode']==country][
            ['date', field]].dropna().reset_index(drop=True)

    def _debug_case_death_ascertainment_plot(self):
        n = 50
        data = self.epidemiology_series[
            ['countrycode', 'days_since_t0', 'date', 'confirmed', 'case_death_ascertainment']]
        worst_n_countries = pd.DataFrame.from_records(
            [{'countrycode': i, 'confirmed': data[data['countrycode']==i]['confirmed'].iloc[-1]}
             for i in data['countrycode'].unique()]).sort_values(
            by=['confirmed'], ascending=False).iloc[0:n]['countrycode'].values
        data_t0 = data[(data['days_since_t0'] >= 0) & (data['countrycode'].isin(worst_n_countries))]
        data_date = data[(data['date'] >= datetime.date(2020, 3, 1)) & (data['countrycode'].isin(worst_n_countries))]
        fig, ax = plt.subplots(nrows=2, figsize=(20, 10))
        for country in worst_n_countries:
            if country in self.debug_countries_of_interest:
                ax[0].plot(data_t0[data_t0['countrycode']==country]['days_since_t0'].values,
                           data_t0[data_t0['countrycode']==country]['case_death_ascertainment'].values,
                           linewidth=2, label=country)
                ax[1].plot(data_date[data_date['countrycode']==country]['date'].values,
                           data_date[data_date['countrycode']==country]['case_death_ascertainment'].values,
                           linewidth=2, label=country)
            else:
                ax[0].plot(data_t0[data_t0['countrycode']==country]['days_since_t0'].values,
                           data_t0[data_t0['countrycode']==country]['case_death_ascertainment'].values, color='grey',
                           alpha=0.25, linewidth=1, label='_nolegend_')
                ax[1].plot(data_date[data_date['countrycode']==country]['date'].values,
                           data_date[data_date['countrycode']==country]['case_death_ascertainment'].values, color='grey',
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
        plt.savefig(self.plot_path + 'inverse_cfr.png')
        plt.close('all')
        return
