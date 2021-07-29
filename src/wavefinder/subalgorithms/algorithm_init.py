import numpy as np
from pandas import Series, DataFrame
from scipy.signal import find_peaks

from wavefinder.utils.prominence_updater import ProminenceUpdater


def init_country(data: Series) -> DataFrame:
    # identify initial list of peaks via find_peaks
    peak = find_peaks(data.values, prominence=0, distance=1)
    trough = find_peaks([-x for x in data.values], prominence=0, distance=1)

    # collect into a single dataframe
    df = DataFrame(data=np.transpose([np.append(data.index[peak[0]], data.index[trough[0]]),
                                      np.append(peak[1]['prominences'], trough[1]['prominences'])]),
                   columns=['location', 'prominence'])
    df['peak_ind'] = np.append([1] * len(peak[0]), [0] * len(trough[0]))
    df.loc[:, 'y_position'] = data[df['location']].values
    df = df.sort_values(by='location').reset_index(drop=True)
    return df

def run(data) -> (DataFrame, ProminenceUpdater):
    if len(data) == 0:
        return data, None, None
    pre_algo = init_country(data)
    prominence_updater = ProminenceUpdater(data)
    return pre_algo, prominence_updater
