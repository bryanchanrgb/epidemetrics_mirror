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
    def apply(data: DataFrame, input_data_df: DataFrame, field: str, prominence_updater: ProminenceUpdater, config: Config) -> DataFrame:

        sub_b_flag = True
        # avoid overwriting sub_a when values replaced by t0 and t1
        df = input_data_df.copy()
        # dictionary to hold boundaries for peak-trough pairs too close to each other
        og_dict = dict()
        while sub_b_flag:
            if len(df) < 2:
                break
            # separation here refers to temporal distance S_i
            df.loc[0:len(df) - 2, 'separation'] = np.diff(df['location'])
            # compute vertical distance V_i
            df.loc[0:len(df) - 2, 'y_distance'] = [abs(x) for x in np.diff(df['y_position'])]
            # sort in ascending order of height
            df = df.sort_values(by='y_distance').reset_index(drop=False)
            # set to false until we find an instance where S_i < t_sep_a / 2 and V_i > v_sep_b
            sub_b_flag = False
            for x in df.index:
                if df.loc[x, 'separation'] < config.t_sep_a / 2:
                    sub_b_flag = True
                    i = df.loc[x, 'index']
                    # get original locations and values
                    og_0 = df.loc[df['index'] == i, 'location'].values[0]
                    og_1 = df.loc[df['index'] == i + 1, 'location'].values[0]
                    y_0 = df.loc[df['index'] == i, 'y_position'].values[0]
                    y_1 = df.loc[df['index'] == i + 1, 'y_position'].values[0]
                    # create boundaries t_0 and t_1 around the peak-trough pair
                    t_0 = max(np.floor((og_0 + og_1 - config.t_sep_a) / 2), 0)
                    t_1 = min(np.floor((og_0 + og_1 + config.t_sep_a) / 2), data[field].index[-1])
                    # store the original locations and values to restore them at the end
                    og_dict[len(og_dict)] = [og_0, t_0, og_1, t_1, y_0, y_1]
                    # reset the peak locations to the boundaries to be rechecked
                    df.loc[df['index'] == i, 'location'] = t_0
                    df.loc[df['index'] == i + 1, 'location'] = t_1
                    df.loc[df['index'] == i, 'y_position'] = data[field].iloc[int(t_0)]
                    df.loc[df['index'] == i + 1, 'y_position'] = data[field].values[int(t_1)]
                    # run the resulting peaks for a prominence check
                    df = prominence_updater.run(df)
                    break

        # restore old locations and heights
        for g in sorted(og_dict, reverse=True):
            df.loc[df['location'] == og_dict[g][1], 'location'] = og_dict[g][0]
            df.loc[df['location'] == og_dict[g][1], 'y_position'] = og_dict[g][4]
            df.loc[df['location'] == og_dict[g][3], 'location'] = og_dict[g][2]
            df.loc[df['location'] == og_dict[g][3], 'y_position'] = og_dict[g][5]
        # recalculate prominence
        df = prominence_updater.run(df)

        return df

    def run(self, input_data_df: DataFrame, country: str, field: str = 'new_per_day_smooth', prominence_updater: ProminenceUpdater = None, plot: bool = False) -> DataFrame:
        data = self.data_provider.get_series(country, field)
        output_data_df = self.apply(data, input_data_df, field, prominence_updater, self.config)
        if plot:
            self.plot(data, input_data_df, output_data_df, field)
        return output_data_df


    def plot(self, data: DataFrame, after_sub_a: DataFrame, after_sub_b: DataFrame, field: str):
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex='all')
        # plot peaks-trough pairs from sub_a
        ax0.set_title('After Sub Algorithm A')
        ax0.plot(data[field].values)
        ax0.scatter(after_sub_a['location'].values,
                    data[field].values[after_sub_a['location'].values.astype(int)], color='red', marker='o')
        # plot peaks-trough pairs from sub_b
        ax1.set_title('After Sub Algorithm B')
        ax1.plot(data[field].values)
        ax1.scatter(after_sub_b['location'].values,
                    data[field].values[after_sub_b['location'].values.astype(int)], color='red', marker='o')
        plt.show()
