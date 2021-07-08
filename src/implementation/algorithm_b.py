import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from data_provider import DataProvider
from implementation.config import Config
from implementation.prominence_updater import ProminenceUpdater


class AlgorithmB:
    def __init__(self, config: Config, data_provider: DataProvider) -> DataFrame:
        self.config = config
        self.data_provider = data_provider

    @staticmethod
    def apply(data: DataFrame, sub_a: DataFrame, field: str, prominence_updater: ProminenceUpdater, config: Config) -> DataFrame:

        sub_b_flag = True
        # avoid overwriting sub_a when values replaced by t0 and t1
        results = sub_a.copy()
        # dictionary to hold boundaries for peak-trough pairs too close to each other
        og_dict = dict()
        while sub_b_flag:
            if len(results) < 2:
                break
            # separation here refers to temporal distance S_i
            results.loc[0:len(results) - 2, 'separation'] = np.diff(results['location'])
            # compute vertical distance V_i
            results.loc[0:len(results) - 2, 'y_distance'] = [abs(x) for x in np.diff(results['y_position'])]
            # sort in ascending order of height
            results = results.sort_values(by='y_distance').reset_index(drop=False)
            # set to false until we find an instance where S_i < t_sep_a / 2 and V_i > v_sep_b
            sub_b_flag = False
            for x in results.index:
                if results.loc[x, 'separation'] < config.t_sep_a / 2:
                    sub_b_flag = True
                    i = results.loc[x, 'index']
                    # get original locations and values
                    og_0 = results.loc[results['index'] == i, 'location'].values[0]
                    og_1 = results.loc[results['index'] == i + 1, 'location'].values[0]
                    y_0 = results.loc[results['index'] == i, 'y_position'].values[0]
                    y_1 = results.loc[results['index'] == i + 1, 'y_position'].values[0]
                    # create boundaries t_0 and t_1 around the peak-trough pair
                    t_0 = max(np.floor((og_0 + og_1 - config.t_sep_a) / 2), 0)
                    t_1 = min(np.floor((og_0 + og_1 + config.t_sep_a) / 2), data[field].index[-1])
                    # store the original locations and values to restore them at the end
                    og_dict[len(og_dict)] = [og_0, t_0, og_1, t_1, y_0, y_1]
                    # reset the peak locations to the boundaries to be rechecked
                    results.loc[results['index'] == i, 'location'] = t_0
                    results.loc[results['index'] == i + 1, 'location'] = t_1
                    results.loc[results['index'] == i, 'y_position'] = data[field].iloc[int(t_0)]
                    results.loc[results['index'] == i + 1, 'y_position'] = data[field].values[int(t_1)]
                    # run the resulting peaks for a prominence check
                    results = prominence_updater.run(results)
                    break

        # restore old locations and heights
        for g in sorted(og_dict, reverse=True):
            results.loc[results['location'] == og_dict[g][1], 'location'] = og_dict[g][0]
            results.loc[results['location'] == og_dict[g][1], 'y_position'] = og_dict[g][4]
            results.loc[results['location'] == og_dict[g][3], 'location'] = og_dict[g][2]
            results.loc[results['location'] == og_dict[g][3], 'y_position'] = og_dict[g][5]
        # recalculate prominence
        results = prominence_updater.run(results)

        return results

    def run(self, sub_a: DataFrame, country: str, field: str = 'new_per_day_smooth', prominence_updater: ProminenceUpdater = None, plot: bool = False) -> DataFrame:
        data = self.data_provider.get_series(country, field)
        results = self.apply(data, sub_a, field, prominence_updater, self.config)
        if plot:
            self.plot(data, sub_a, results, field)
        return results

    def plot(self, data: DataFrame, sub_a: DataFrame, results: DataFrame, field: str):
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
        plt.show()
