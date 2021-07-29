"""
NAME
    algorithm_b

DESCRIPTION
    This module provides Sub-Algorithm B to WaveList in order to merge short transient features.

FUNCTIONS
    run
"""

import numpy as np
from pandas import DataFrame, Series

from wavefinder.utils.prominence_updater import ProminenceUpdater


def run(raw_data: Series, input_data_df: DataFrame,
        prominence_updater: ProminenceUpdater, t_sep_a: int) -> DataFrame:
    """
    Identifies pairs of minima and maxima separated by less than t_sep_a/2 and merges them if they are transient

    Parameters:
        raw_data (Series): The original data from which the peaks and troughs are identified
        input_data_df (DataFrame): The list of peaks and troughs to be merged.
        prominence_updater (ProminenceUpdater): An object to recalculate prominence of peaks and troughs after each
        deletion.
        t_sep_a (int): Threshold specifying which features should be investigated.

    Returns:
        run(raw_data, input_data_df, prominence_updater, t_sep_a): The list of peaks and troughs after merging.
    """

    # flag will be dropped once no pair is found
    sub_b_flag = True
    # avoid overwriting sub_a when values replaced by t0 and t1
    df = input_data_df.copy()
    # dictionary to hold boundaries for peak-trough pairs too close to each other
    original_data = dict()
    while sub_b_flag:
        if len(df) < 2:
            break
        # separation here refers to temporal distance S_i
        df.loc[0:len(df) - 2, 'separation'] = np.diff(df['location'])
        # compute vertical distance V_i
        df.loc[0:len(df) - 2, 'y_distance'] = [abs(x) for x in np.diff(df['y_position'])]
        # sort in ascending order of height
        df = df.sort_values(by='y_distance').reset_index(drop=False)
        # set to false until we find an instance where S_i < t_sep_a / 2
        sub_b_flag = False
        for x in df.index:
            if df.loc[x, 'separation'] < t_sep_a / 2:
                sub_b_flag = True
                i = df.loc[x, 'index']
                # store the original locations and values to restore them at the end
                original_t_0 = df.loc[df['index'] == i, 'location'].values[0]
                original_t_1 = df.loc[df['index'] == i + 1, 'location'].values[0]
                y_0 = df.loc[df['index'] == i, 'y_position'].values[0]
                y_1 = df.loc[df['index'] == i + 1, 'y_position'].values[0]
                # create boundaries t_0 and t_1 around the peak-trough pair
                t_0 = max(np.floor((original_t_0 + original_t_1 - t_sep_a) / 2), 0)
                t_1 = min(np.floor((original_t_0 + original_t_1 + t_sep_a) / 2), raw_data.index[-1])
                # add original locations and value to dictionary
                original_data[t_0] = {'location': original_t_0, 'y_position': y_0}
                original_data[t_1] = {'location': original_t_1, 'y_position': y_1}
                # reset the peak locations to the boundaries to be rechecked
                df.loc[df['index'] == i, 'location'] = t_0
                df.loc[df['index'] == i + 1, 'location'] = t_1
                df.loc[df['index'] == i, 'y_position'] = raw_data.iloc[int(t_0)]
                df.loc[df['index'] == i + 1, 'y_position'] = raw_data.values[int(t_1)]
                # run the resulting peaks for a prominence check
                df = prominence_updater.run(df)
                break

    # restore old locations and heights
    for key, val in original_data.items():
        df.loc[df['location'] == key, ['y_position', 'location']] = [val['y_position'], val['location']]

    # recalculate prominence
    df = prominence_updater.run(df)

    return df
