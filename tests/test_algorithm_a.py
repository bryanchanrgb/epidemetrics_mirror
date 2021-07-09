import pytest
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from pandas import DataFrame
from typing import List

from implementation.config import Config
from implementation.algorithm_a import AlgorithmA
from data_provider import ListDataProvider
from implementation.algorithm_init import AlgorithmInit
from implementation.prominence_updater import ProminenceUpdater
from plot_helper import plot_results


class TestAlgorithmA:

    @classmethod
    def setup_class(cls):
        cls.config = Config()
        cls.country = 'TEST'
        cls.field = 'new_per_day_smooth'

    def test_1(self):
        input_data = [1, 10, 5, 7, 6, 20, 19]

        data_provider = ListDataProvider(input_data, self.country, self.field, x_scaling_factor=7)

        data = data_provider.get_series(self.country, self.field)
        peaks_initial = AlgorithmInit(None, None).init_country(data[self.field])
        prominence_updater = ProminenceUpdater(data, self.field)

        result = AlgorithmA(self.config).run(peaks_initial, prominence_updater)

        plot_results(raw_data=data[self.field], peaks_before=peaks_initial, peaks_after=result)
        y_positions = result["y_position"].to_list()

        expected_result = [10, 5, 20]
        assert y_positions == expected_result
