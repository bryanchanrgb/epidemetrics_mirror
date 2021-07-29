"""
NAME
    algorithm_c_and_d

DESCRIPTION
    This module provides Sub-Algorithms C and D to WaveList in order to merge waves of low prominence.

FUNCTIONS
    run
"""

from pandas import DataFrame, Series

import wavefinder.utils.trough_finder as trough_finder


def run(raw_data: Series, input_data_df: DataFrame, prominence_threshold: float,
        proportional_prominence_threshold: float) -> DataFrame:
    """
    Identifies waves with low prominence and merges them

    Parameters:
        raw_data (Series): The original data from which the peaks and troughs are identified
        input_data_df (DataFrame): The list of peaks and troughs to be merged.
        prominence_threshold (float): The minimum prominence which a wave must have.
        proportional_prominence_threshold (float): The minimum prominence which a peak must have, as a ratio of the
        value at the peak.

    Returns:
        run(raw_data, input_data_df, prominence_threshold, proportional_prominence_threshold): The list of peaks and
        troughs after merging.
    """

    df = input_data_df.copy()
    df = df.sort_values(by='location').reset_index(drop=True)
    # filter out troughs and peaks below prominence threshold
    peaks = df[df['peak_ind'] == 1].reset_index(drop=True)
    troughs = df[df['peak_ind'] == 0].reset_index(drop=True)

    # filter out a peak and its corresponding trough if the peak does not meet the prominence threshold
    peaks_c = peaks[peaks['prominence'] >= prominence_threshold]
    # filter out relatively low prominent peaks
    peaks_d = peaks_c[
        (peaks_c['prominence'] >= proportional_prominence_threshold * peaks_c['y_position'])]
    # between each remaining peak, retain the trough with the lowest value
    df = trough_finder.run(peaks_d, troughs, raw_data, prominence_threshold, proportional_prominence_threshold)

    return df
