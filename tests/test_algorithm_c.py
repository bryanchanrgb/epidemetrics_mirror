import pytest
import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List

from implementation.config import Config
from implementation.algorithm_a import AlgorithmA
from implementation.algorithm_b import AlgorithmB
from implementation.algorithm_c import AlgorithmC
from data_provider import ListDataProvider
from implementation.pre_algo import PreAlgo
from implementation.prominence_updater import ProminenceUpdater
from plot_helper import plot_results


class TestAlgorithmC:

    @classmethod
    def setup_class(cls):
        cls.config = Config()
        cls.country = 'TEST'
        cls.field = 'new_per_day_smooth'

    def test_1(self):
        input_data = [10, 80, 20, 60, 10, 80, 30, 110, 25]

        data_provider = ListDataProvider(input_data, self.country, self.field, x_scaling_factor=14)

        pre_algo = PreAlgo(self.config, data_provider)
        data, peaks_initial = pre_algo.init_country(self.country, self.field)
        prominence_updater = ProminenceUpdater(data, self.field)

        sub_a = AlgorithmA(self.config).run(peaks_initial, prominence_updater)

        sub_b = AlgorithmB(self.config).run(
            raw_data=data[self.field],
            input_data_df=sub_a,
            prominence_updater=prominence_updater)

        result = AlgorithmC(self.config, data_provider, self.country, field=self.field).run(
            raw_data=data[self.field],
            input_data_df=sub_b,
        )

        plot_results(raw_data=data[self.field], peaks_before=sub_b, peaks_after=result)

        y_positions = result["y_position"].to_list()

        expected_result = [80, 10, 110]
        assert y_positions == expected_result
