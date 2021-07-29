import os
from pandas import DataFrame
import matplotlib.pyplot as plt

from wavefinder.wavelist import WaveList


def plot_cross_validator(input_wavelist: WaveList, reference_wavelist: WaveList, filename: str, results: DataFrame,
                         plot_path: str):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    # plot peaks after sub_c
    axs[0, 0].set_title('Peaks in Original Series')
    axs[0, 0].plot(input_wavelist.raw_data.values)
    axs[0, 0].scatter(input_wavelist.peaks_sub_c['location'].values,
                      input_wavelist.raw_data.values[
                          input_wavelist.peaks_sub_c['location'].values.astype(int)], color='red', marker='o')
    # plot peaks from sub_e
    axs[0, 1].set_title('After Cross-Validation')
    axs[0, 1].plot(input_wavelist.raw_data.values)
    axs[0, 1].scatter(results['location'].values,
                      input_wavelist.raw_data.values[
                          results['location'].values.astype(int)], color='red', marker='o')
    # plot peaks from reference series
    axs[1, 1].set_title('Peaks in Reference Series')
    axs[1, 1].plot(reference_wavelist.raw_data.values)
    axs[1, 1].scatter(reference_wavelist.peaks_sub_c['location'].values,
                      reference_wavelist.raw_data.values[
                          reference_wavelist.peaks_sub_c['location'].values.astype(int)], color='red', marker='o')

    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, filename + '_algorithm_e.png'))
    plt.close('all')


def plot_peaks(wavelists: list, title: str, plot_path: str, save: bool):
    # if a single WaveList is passed, package it in a list so the method works
    if isinstance(wavelists, WaveList):
        wavelists = [wavelists]

    columns = [{'desc': ' Before Algorithm', 'source': 'peaks_initial'},
               {'desc': ' After Sub Algorithm A', 'source': 'peaks_sub_a'},
               {'desc': ' After Sub Algorithm B', 'source': 'peaks_sub_b'},
               {'desc': ' After Sub Algorithm C&D', 'source': 'peaks_sub_c'}]

    fig, axs = plt.subplots(nrows=len(wavelists), ncols=len(columns), sharex=True, figsize=(14, 7))
    plt.suptitle(title)

    for i, wavelist in enumerate(wavelists):
        for j, column in enumerate(columns):
            peaks = getattr(wavelist, column['source'])['location'].values
            axs[i, j].set_title(wavelist.series_name + column['desc'])
            axs[i, j].plot(wavelist.raw_data.values)
            axs[i, j].scatter(peaks, wavelist.raw_data.values[peaks.astype(int)], color='red', marker='o')
            axs[i, j].get_xaxis().set_visible(False)
            axs[i, j].get_yaxis().set_visible(False)

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(plot_path, title + '.png'))
        plt.close('all')
