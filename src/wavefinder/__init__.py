"""
NAME
    wavefinder

DESCRIPTION
    A Python package to identify waves in time series.
    ==================================================

    wavefinder provides two classes and two associated plotting functions. WaveList implements an algorithm to
    identify the waves in time series, which can be plotted using plot_peaks. It also implements an
    algorithm to impute additional waves from a reference WaveList object,
    which plot_cross_validator plots.

PACKAGE CONTENTS
    WaveList
    plot_peaks
    plot_cross_validator
"""

from wavefinder.wavelist import WaveList
from wavefinder.waveplotter import plot_peaks, plot_cross_validator

__all__ = ['WaveList', 'plot_peaks', 'plot_cross_validator']
