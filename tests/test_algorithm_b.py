import pytest
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List

from implementation.config import Config
from implementation.algorithm_a import AlgorithmA
from implementation.algorithm_b import AlgorithmB
from data_provider import ListDataProvider
from implementation.pre_algo import PreAlgo
from implementation.prominence_updater import ProminenceUpdater
from plot_helper import plot_results


class TestAlgorithmB:

    @classmethod
    def setup_class(cls):
        cls.config = Config()
        cls.country = 'TEST'
        cls.field = 'new_per_day_smooth'

    def test_1(self):
        input_data = [1, 10, 5, 15, 20, 10]

        data_provider = ListDataProvider(input_data, self.country, self.field, x_scaling_factor=7)

        pre_algo = PreAlgo(self.config, data_provider)
        data, peaks_initial = pre_algo.init_country(self.country, self.field)
        prominence_updater = ProminenceUpdater(data, self.field)

        sub_a = AlgorithmA(self.config).run(peaks_initial, prominence_updater)

        result = AlgorithmB(self.config).run(raw_data=data[self.field],
                                             input_data_df=sub_a,
                                             prominence_updater=prominence_updater)

        plot_results(raw_data=data[self.field], peaks_before=sub_a, peaks_after=result)

        y_positions = result["y_position"].to_list()

        expected_result = [20]
        assert y_positions == expected_result
