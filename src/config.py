import os
from dataclasses import dataclass, field


@dataclass
class Config:
    rel_to_constant = 10000  # for calculating relative to the population

    # for waves algorithm
    abs_prominence_threshold = 45  # minimum prominence
    rel_prominence_threshold = 0.033  # prominence relative to rel_to_constant
    rel_prominence_max_threshold = 500  # upper limit on relative prominence
    prominence_height_threshold = 0.61  # prominence must be above a percentage of the peak height
    abs_prominence_threshold_dead = 7  # minimum prominence for dead peak detection
    rel_prominence_threshold_dead = 0.01  # prominence threshold for dead rel_to_constant
    rel_prominence_max_threshold_dead = 70  # upper limit on relative prominencce
    prominence_height_threshold_dead = 0.65  # prominence must be above a percentage of the peak height
    t_sep_a = 35

    # for analysis
    abs_t0_threshold = 1000
    rel_t0_threshold = 0.05  # cases per rel_to_constant
    exclude_countries = []  # countries with low quality data to be ignored
    class_1_threshold = 55  # minimum number of absolute cases to be considered going into first wave
    class_1_threshold_dead = 5
    debug_death_lag = 9  # death lag for case-death ascertainment
    debug_countries_of_interest = ['USA', 'GBR', 'BRA', 'IND', 'ESP', 'FRA', 'ZAF']

    # for storage
    base_path: str = None
    plot_path: str = field(init=False)
    data_path: str = field(init=False)
    cache_path: str = field(init=False)

    def __post_init__(self):
        if not self.base_path:
            return

        self.plot_path = os.path.abspath(os.path.join(self.base_path, '../plots/algorithm_results'))
        self.data_path = os.path.abspath(os.path.join(self.base_path, '../data'))
        self.cache_path = os.path.abspath(os.path.join(self.base_path, '../cache'))

    def prominence_thresholds(self, field):
        if field == 'new_per_day_smooth':
            thresholds = {"abs_prominence_threshold": self.abs_prominence_threshold,
                          "rel_prominence_threshold": self.rel_prominence_threshold,
                          "rel_prominence_max_threshold": self.rel_prominence_max_threshold,
                          "prominence_height_threshold": self.prominence_height_threshold}
            return thresholds
        elif field == 'dead_per_day_smooth':
            thresholds = {"abs_prominence_threshold": self.abs_prominence_threshold_dead,
                          "rel_prominence_threshold": self.rel_prominence_threshold_dead,
                          "rel_prominence_max_threshold": self.rel_prominence_max_threshold_dead,
                          "prominence_height_threshold": self.prominence_height_threshold_dead}
            return thresholds
        else:
            return None
