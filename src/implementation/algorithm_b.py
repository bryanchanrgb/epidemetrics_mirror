import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from data_provider import DataProvider
from implementation.config import Config
from implementation.algorithm_a import AlgorithmA


class AlgorithmB:
    def __init__(self, config: Config, data_provider: DataProvider, algorithm_a: AlgorithmA) -> DataFrame:
        self.config = config
        self.data_provider = data_provider
        self.algorithm_a = algorithm_a

    def run(self, sub_a, country, field='new_per_day_smooth', plot=False):
        data = self.data_provider.get_series(country, field)
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
                if results.loc[x, 'y_distance'] >= self.config.v_sep_b and results.loc[
                    x, 'separation'] < self.config.t_sep_a / 2:
                    sub_b_flag = True
                    i = results.loc[x, 'index']
                    og_0 = results.loc[results['index'] == i, 'location'].values[0]
                    og_1 = results.loc[results['index'] == i + 1, 'location'].values[0]
                    # creating boundaries t_0 and t_1 around the peak-trough pair
                    t_0 = np.floor((og_0 + og_1 - self.config.t_sep_a) / 2)
                    t_1 = np.floor((og_0 + og_1 + self.config.t_sep_a) / 2)
                    # store the original locations to restore them at the end
                    og_dict[len(og_dict)] = [og_0, t_0, og_1, t_1]
                    # setting the peak locations to the boundaries to be filtered by sub_algorithm_a
                    results.loc[results['index'] == i, 'location'] = t_0
                    results.loc[results['index'] == i + 1, 'location'] = t_1
                    # run the indices list (adding start and end of the time series to the list) through find_peaks again
                    locations = np.clip(np.sort(np.append(results['location'],
                                                          [min(data[field][~np.isnan(data[field].values)].index),
                                                           max(data[field][~np.isnan(data[field])].index)])), 0,
                                        len(data) - 1)
                    # run the resulting peaks through sub algorithm A again
                    results = self.algorithm_a.run(country=country, field=field,
                                                   plot=False, override=data.iloc[locations])
                    break

        for g in sorted(og_dict, reverse=True):
            results.loc[results['location'] == og_dict[g][1], 'location'] = og_dict[g][0]
            results.loc[results['location'] == og_dict[g][3], 'location'] = og_dict[g][2]

        if plot:
            self.plot(data, sub_a, results, field)
        return results

    def plot(self, data, sub_a, results, field):
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
