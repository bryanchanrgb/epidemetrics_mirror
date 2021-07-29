import numpy as np
from pandas import DataFrame

from wavefinder.utils.prominence_updater import ProminenceUpdater


def delete_pairs(data: DataFrame, t_sep_a: int) -> DataFrame:
    if np.nanmin(data['duration']) < t_sep_a and len(data) >= 3:
        # extract waves of low duration
        df1 = data[data['duration'] < t_sep_a]
        # extract those of least prominence
        df2 = df1[df1['prominence'] == min(df1['prominence'])]
        # find the shortest
        i = df2.idxmin()['duration']
        is_peak = data.loc[i, 'peak_ind'] - 0.5
        data.drop(index=i, inplace=True)
        # remove whichever adjacent candidate is a greater minimum, or a lesser maximum. If tied, remove the
        # earlier.
        if is_peak * (data.loc[i + 1, 'y_position'] - data.loc[i - 1, 'y_position']) >= 0:
            data.drop(index=i + 1, inplace=True)
        else:
            data.drop(index=i - 1, inplace=True)
    return data


def run(input_data_df: DataFrame, prominence_updater: ProminenceUpdater, t_sep_a: int) -> DataFrame:
    df = input_data_df.copy()
    # if there are fewer than 3 points, the algorithm cannot be run, return list unchanged
    if len(df) < 3:
        return df

    else:
        # calculate the duration of extremum i as the distance between the extrema to the left and to the right of i
        df['duration'] = df['location'].diff(periods=1) - df['location'].diff(periods=-1)

    while np.nanmin(df['duration']) < t_sep_a and len(df) >= 3:
        # remove peaks and troughs until the smallest duration meets T_SEP
        df = delete_pairs(df, t_sep_a)
        # update prominence
        df = prominence_updater.run(df)
        # recalculate duration
        if len(df) >= 3:
            df['duration'] = df['location'].diff(periods=1) - df['location'].diff(periods=-1)
        else:
            break

    # results returns a set of peaks and troughs which are at least a minimum distance apart
    return df
