import os
from pandas import DataFrame
import matplotlib.pyplot as plt


def plot_results(raw_data: DataFrame, peaks_before: DataFrame, peaks_after: DataFrame):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex='all')
    # plot peaks-trough pairs from sub_a
    ax0.set_title('Data before Algorithm')
    ax0.plot(raw_data.values)
    ax0.scatter(peaks_before['location'].values,
                raw_data.values[peaks_before['location'].values.astype(int)], color='red', marker='o')
    # plot peaks-trough pairs from sub_b
    ax1.set_title('Data after Algorithm')
    ax1.plot(raw_data.values)
    ax1.scatter(peaks_after['location'].values,
                raw_data.values[peaks_after['location'].values.astype(int)], color='red', marker='o')
    plt.show()


def plot_e(data: DataFrame, country: str, cases_sub_c: DataFrame, results: DataFrame,
           deaths_data: DataFrame, deaths_sub_c: DataFrame, plot_path: str):
    fig, axs = plt.subplots(nrows=2, ncols=2)
    # plot peaks after sub_c
    axs[0, 0].set_title('After Sub Algorithm C & D')
    axs[0, 0].plot(data['new_per_day_smooth'].values)
    axs[0, 0].scatter(cases_sub_c['location'].values,
                      data['new_per_day_smooth'].values[
                          cases_sub_c['location'].values.astype(int)], color='red', marker='o')
    # plot peaks from sub_e
    axs[0, 1].set_title('After Sub Algorithm E')
    axs[0, 1].plot(data['new_per_day_smooth'].values)
    axs[0, 1].scatter(results['location'].values,
                      data['new_per_day_smooth'].values[
                          results['location'].values.astype(int)], color='red', marker='o')
    # plot death peaks
    axs[1, 1].set_title('Death Peaks')
    axs[1, 1].plot(deaths_data['dead_per_day_smooth'].values)
    axs[1, 1].scatter(deaths_sub_c['location'].values,
                      deaths_data['dead_per_day_smooth'].values[
                          deaths_sub_c['location'].values.astype(int)], color='red', marker='o')

    fig.tight_layout()
    plt.savefig(os.path.join(plot_path, country + '_algorithm_e.png'))
    plt.close('all')


def plot_peaks(cases, deaths, country, cases_initial, cases_cleaned, cases_sub_a, cases_sub_b, cases_sub_c,
               deaths_initial, deaths_cleaned, deaths_sub_a, deaths_sub_b, deaths_sub_c, plot_path, save):
    fig, axs = plt.subplots(nrows=2, ncols=5, sharex=True, figsize=(14, 7))
    plt.suptitle(country)

    axs[0, 0].set_title('Cases Before Algorithm')
    axs[0, 0].plot(cases['new_per_day_smooth'].values)
    axs[0, 0].scatter(cases_initial['location'].values,
                      cases['new_per_day_smooth'].values[
                          cases_initial['location'].values.astype(int)], color='red', marker='o')
    axs[0, 0].get_xaxis().set_visible(False)
    axs[0, 0].get_yaxis().set_visible(False)

    axs[0, 1].set_title('Cases After Spike Removal')
    axs[0, 1].plot(cases['new_per_day_smooth'].values)
    axs[0, 1].scatter(cases_cleaned['location'].values,
                      cases['new_per_day_smooth'].values[
                          cases_cleaned['location'].values.astype(int)], color='red', marker='o')
    axs[0, 2].get_xaxis().set_visible(False)
    axs[0, 2].get_yaxis().set_visible(False)

    axs[0, 2].set_title('Cases After Sub Algorithm A')
    axs[0, 2].plot(cases['new_per_day_smooth'].values)
    axs[0, 2].scatter(cases_sub_a['location'].values,
                      cases['new_per_day_smooth'].values[
                          cases_sub_a['location'].values.astype(int)], color='red', marker='o')
    axs[0, 2].get_xaxis().set_visible(False)
    axs[0, 2].get_yaxis().set_visible(False)

    axs[0, 3].set_title('Cases After Sub Algorithm B')
    axs[0, 3].plot(cases['new_per_day_smooth'].values)
    axs[0, 3].scatter(cases_sub_b['location'].values,
                      cases['new_per_day_smooth'].values[
                          cases_sub_b['location'].values.astype(int)], color='red', marker='o')
    axs[0, 3].get_xaxis().set_visible(False)
    axs[0, 3].get_yaxis().set_visible(False)

    axs[0, 4].set_title('Cases After Sub Algorithm C&D')
    axs[0, 4].plot(cases['new_per_day_smooth'].values)
    axs[0, 4].scatter(cases_sub_c['location'].values,
                      cases['new_per_day_smooth'].values[
                          cases_sub_c['location'].values.astype(int)], color='red', marker='o')
    axs[0, 4].get_xaxis().set_visible(False)
    axs[0, 4].get_yaxis().set_visible(False)

    axs[1, 0].set_title('Deaths Before Algorithm')
    axs[1, 0].plot(deaths['dead_per_day_smooth'].values)
    axs[1, 0].scatter(deaths_initial['location'].values,
                      deaths['dead_per_day_smooth'].values[
                          deaths_initial['location'].values.astype(int)], color='red', marker='o')
    axs[1, 0].get_xaxis().set_visible(False)
    axs[1, 0].get_yaxis().set_visible(False)

    axs[1, 1].set_title('Deaths After Spike Removal')
    axs[1, 1].plot(deaths['dead_per_day_smooth'].values)
    axs[1, 1].scatter(deaths_cleaned['location'].values,
                      deaths['dead_per_day_smooth'].values[
                          deaths_cleaned['location'].values.astype(int)], color='red', marker='o')
    axs[1, 1].get_xaxis().set_visible(False)
    axs[1, 1].get_yaxis().set_visible(False)

    axs[1, 2].set_title('Deaths After Sub Algorithm A')
    axs[1, 2].plot(deaths['dead_per_day_smooth'].values)
    axs[1, 2].scatter(deaths_sub_a['location'].values,
                      deaths['dead_per_day_smooth'].values[
                          deaths_sub_a['location'].values.astype(int)], color='red', marker='o')
    axs[1, 2].get_xaxis().set_visible(False)
    axs[1, 2].get_yaxis().set_visible(False)

    axs[1, 3].set_title('Deaths After Sub Algorithm B')
    axs[1, 3].plot(deaths['dead_per_day_smooth'].values)
    axs[1, 3].scatter(deaths_sub_b['location'].values,
                      deaths['dead_per_day_smooth'].values[
                          deaths_sub_b['location'].values.astype(int)], color='red', marker='o')
    axs[1, 3].get_xaxis().set_visible(False)
    axs[1, 3].get_yaxis().set_visible(False)

    axs[1, 4].set_title('Deaths After Sub Algorithm C&D')
    axs[1, 4].plot(deaths['dead_per_day_smooth'].values)
    axs[1, 4].scatter(deaths_sub_c['location'].values,
                      deaths['dead_per_day_smooth'].values[
                          deaths_sub_c['location'].values.astype(int)], color='red', marker='o')
    axs[1, 4].get_xaxis().set_visible(False)
    axs[1, 4].get_yaxis().set_visible(False)

    fig.tight_layout()

    if save:
        plt.savefig(os.path.join(plot_path, country + '.png'))
        plt.close('all')
