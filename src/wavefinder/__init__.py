"""
NAME
    wavefinder

DESCRIPTION
    A Python package to identify waves in time series.
    ==================================================

    wavefinder provides two classes and two associated plotting functions. WaveList implements an algorithm to
    identify the waves in time series, which can be plotted using plot_peaks. WaveCrossValidator implements an
    algorithm to impute additional waves in one WaveList object using a reference WaveList object,
    which plot_cross_validator plots.

PACKAGE CONTENTS
    WaveList
    WaveCrossValidator
    plot_peaks
    plot_cross_validator
"""

from wavefinder.wavelist import WaveList
from wavefinder.wavecrossvalidator import WaveCrossValidator
from wavefinder.waveplotter import plot_peaks, plot_cross_validator

__all__ = ['WaveList', 'WaveCrossValidator', 'plot_peaks', 'plot_cross_validator']
