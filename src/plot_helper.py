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
