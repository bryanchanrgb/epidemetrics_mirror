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
        input_data = [1, 5, 3, 10, 20, 19]

        data_provider = ListDataProvider(input_data, x_scaling_factor=7)

        algorithm_a = AlgorithmA(self.config,
                                 data_provider=data_provider)
        algorithm_b = AlgorithmB(self.config, algorithm_a=algorithm_a,
                                 data_provider=data_provider)
        algorithm_c = AlgorithmC(self.config, data_provider=data_provider)

        sub_a = algorithm_a.run(country='TEST', field='value', plot=True)
        sub_b = algorithm_b.run(sub_a, country='TEST', field='value', plot=True)
        result = algorithm_c.run(sub_a, sub_b, country='TEST', field='value', plot=True)
        y_positions = result["y_position"].to_list()

        expected_result = []
        assert y_positions == expected_result
