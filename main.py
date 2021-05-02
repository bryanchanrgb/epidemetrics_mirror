import numpy as np
import pandas as pd
import datetime
from tqdm import tqdm
from csaps import csaps
import psycopg2
import matplotlib.pyplot as plt

class epidemetrics:
    def __init__(self):
        self.source = 'WRD_WHO'
        self.end_date = datetime.date(2021, 4, 1)
        self.ma_window = 7
        self.use_splines = False
        self.smooth = 0.001
        self.flags = {'c1_school_closing': 3,
                      'c2_workplace_closing': 3,
                      'c3_cancel_public_events': 2,
                      'c4_restrictions_on_gatherings': 4,
                      'c5_close_public_transport': 2,
                      'c6_stay_at_home_requirements': 2,
                      'c7_restrictions_on_internal_movement': 2,
                      'c8_international_travel_controls': 4,
                      'h2_testing_policy': 3,
                      'h3_contact_tracing': 2}
        self.wb_codes = {
            'SP.POP.TOTL': 'value',
            'EN.POP.DNST': 'population_density',
            'NY.GNP.PCAP.PP.KD': 'gni_per_capita',
            'SM.POP.NETM': 'net_migration'}
        self.abs_t0_threshold = 1000
        self.rel_t0_threshold = 0.05 # cases per rel_to_constant
        self.rel_to_constant = 10000 # used as population reference for relative t0

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
            'days_since_t0_10_dead': np.empty(0)
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
            continue
        return pd.DataFrame.from_dict(epidemiology_series)

    def _get_epi_static(self):
        return

    def _get_gsi_table(self):
        '''
        PREPARE GOVERNMENT RESPONSE TABLE
        '''
        cols = 'countrycode, country, date, stringency_index, ' + ', '.join(list(self.flags.keys()))
        sql_command = """SELECT """ + cols + """ FROM government_response"""
        gsi_table = pd.read_sql(sql_command, self.conn) \
                            .sort_values(by=['countrycode', 'date']) \
                            .filter(items=['countrycode', 'country', 'date', 'stringency_index'] + self.flags) \
                            .reset_index(drop=True)
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

    def _sub_aglorithm_a(self, country):
        return