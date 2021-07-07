import pytest
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from pandas import DataFrame
from typing import List

from implementation.config import Config
from implementation.algorithm_a import AlgorithmA
from data_provider import ListDataProvider


class TestAlgorithmA:

    @classmethod
    def setup_class(cls):
        cls.config = Config()

    def test_1(self):
        input_data = [1, 10, 5, 7, 6, 20, 19]
        field = 'new_per_day_smooth'

        data_provider = ListDataProvider(input_data, field=field, x_scaling_factor=7)

        algorithm_a = AlgorithmA(self.config, data_provider)
        result = algorithm_a.run(country='TEST', field=field, plot=True)
        y_positions = result["y_position"].to_list()

        expected_result = [10, 5, 20]
        assert y_positions == expected_result
