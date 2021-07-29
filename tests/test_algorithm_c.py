from config import Config
from data_provider import ListDataProvider
from wavefinder.utils.prominence_updater import ProminenceUpdater
import wavefinder.subalgorithms.algorithm_init as algorithm_init
import wavefinder.subalgorithms.algorithm_a as algorithm_a
import wavefinder.subalgorithms.algorithm_b as algorithm_b
import wavefinder.subalgorithms.algorithm_c as algorithm_c
from plot_helper import plot_results


class TestAlgorithmC:

    @classmethod
    def setup_class(cls):
        cls.config = Config()
        cls.country = 'TEST'
        cls.field = 'new_per_day_smooth'

    def test_1(self):
        input_data = [10, 80, 20, 60, 10, 80, 30, 110, 25]

        data_provider = ListDataProvider(input_data, self.country, self.field, x_scaling_factor=7)

        data = data_provider.get_series(self.country, self.field)[self.field]
        peaks_initial = algorithm_init.init_country(data)
        prominence_updater = ProminenceUpdater(data)

        params = self.config.prominence_thresholds(self.field)
        params['rel_to_constant'] = self.config.rel_to_constant
        population = data_provider.get_population(self.country)
        prominence_threshold = max(params['abs_prominence_threshold'],
                                   min(params['rel_prominence_threshold'] * population / params['rel_to_constant'],
                                       params['rel_prominence_max_threshold']))
        prominence_height_threshold = params['prominence_height_threshold']

        sub_a = algorithm_a.run(
            input_data_df=peaks_initial,
            prominence_updater=prominence_updater,
            t_sep_a=self.config.t_sep_a)

        sub_b = algorithm_b.run(
            raw_data=data,
            input_data_df=sub_a,
            prominence_updater=prominence_updater,
            t_sep_a=self.config.t_sep_a)

        result = algorithm_c.run(
            raw_data=data,
            input_data_df=sub_b,
            prominence_threshold=prominence_threshold,
            proportional_prominence_threshold=prominence_height_threshold)

        plot_results(raw_data=data, peaks_before=sub_b, peaks_after=result)

        y_positions = result["y_position"].to_list()

        expected_result = [80, 10, 110]
        assert y_positions == expected_result
