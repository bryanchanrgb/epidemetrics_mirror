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


class TestAlgorithmC:

    @classmethod
    def setup_class(cls):
        cls.config = Config()

    def test_1(self):
        input_data = [10, 80, 20, 60, 10, 80, 30, 110, 25]
        field = 'new_per_day_smooth'

        data_provider = ListDataProvider(input_data, field=field, x_scaling_factor=14)

        algorithm_a = AlgorithmA(self.config, data_provider=data_provider)
        algorithm_b = AlgorithmB(self.config, data_provider=data_provider)
        algorithm_c = AlgorithmC(self.config, data_provider=data_provider)

        sub_a = algorithm_a.run(country='TEST', field=field, plot=True)
        sub_b = algorithm_b.run(sub_a, country='TEST', field=field, plot=True)
        result = algorithm_c.run(sub_b, country='TEST', field=field, plot=True)
        y_positions = result["y_position"].to_list()

        expected_result = [80, 10, 110]
        assert y_positions == expected_result
