import pytest
import numpy as np
import pandas as pd
import scipy.interpolate as interp
from pandas import DataFrame
from typing import List

from src.implementation.config import Config
from src.implementation.algorithm_a import AlgorithmA


class ListDataProvider:

    def __init__(self, data: List, country: str = 'TEST'):
        # Stretch input data list to 100 points
        arr_interp = interp.interp1d(np.arange(len(data)), data)
        data_stretch = arr_interp(np.linspace(0, len(data) - 1, 100))

        self.df = pd.DataFrame({'value': data_stretch})
        self.df['countrycode'] = country
        self.df['date'] = pd.date_range(start='1/1/2020', periods=len(self.df), freq='D')

    def get_series(self, country: str, field: str) -> DataFrame:
        return self.df[self.df['countrycode'] == country][['date', field]].dropna().reset_index(drop=True)


class TestAlgorithmA:

    @classmethod
    def setup_class(cls):
        cls.config = Config()

    def test_1(self):
        input_data = [0, 5, 10, 9, 5, 7, 6, 11, 20, 18]

        algorithm_a = AlgorithmA(self.config, data_provider=ListDataProvider(input_data))
        result = algorithm_a.run(country='TEST', field='value', plot=False, override=None)

        # [prominence, location, peak_ind,left_base, right_base, duration]
        expected_result = np.array([5.0, 2.0, 1.0, 0.0, 4.0, np.nan])
        assert np.array_equal(result.values[0], expected_result, equal_nan=True)
