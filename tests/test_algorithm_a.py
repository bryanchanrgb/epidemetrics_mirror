from config import Config
import wavefinder.subalgorithms.algorithm_a as algorithm_a
from data_provider import ListDataProvider
import wavefinder.subalgorithms.algorithm_init as algorithm_init
from wavefinder.utils.prominence_updater import ProminenceUpdater
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

        data = data_provider.get_series(self.country, self.field)[self.field]
        peaks_initial = algorithm_init.init_country(data)
        prominence_updater = ProminenceUpdater(data)

        result = algorithm_a.run(
            input_data_df=peaks_initial,
            prominence_updater=prominence_updater,
            t_sep_a=self.config.t_sep_a)

        plot_results(raw_data=data, peaks_before=peaks_initial, peaks_after=result)
        y_positions = result["y_position"].to_list()

        expected_result = [10, 5, 20]
        assert y_positions == expected_result
