import os
from dataclasses import dataclass, field


@dataclass
class Config:
    spike_sensitivity = 1.5
    abs_t0_threshold = 1000
    rel_t0_threshold = 0.05  # cases per rel_to_constant
    rel_to_constant = 10000  # used as population reference for relative t0

    # for waves algorithm
    abs_prominence_threshold = 50  # minimum prominence
    abs_prominence_threshold_dead = 7  # minimum prominence for dead peak detection
    rel_prominence_threshold = 0.1  # prominence relative to rel_to_constant
    rel_prominence_threshold_dead = 0.014  # prominence threshold for dead rel_to_constant
    rel_prominence_max_threshold = 500  # upper limit on relative prominence
    rel_prominence_max_threshold_dead = 70  # upper limit on relative prominencce
    prominence_height_threshold = 0.5  # prominence must be above a percentage of the peak height
    prominence_height_threshold_dead = 0.5  # prominence must be above a percentage of the peak height
    t_sep_a = 21

    v_sep_b = 1  # v separation for sub algorithm B

    d_match = 35  # matching window for undetected case waves based on death waves
    exclude_countries = ['CMR', 'COG', 'GNQ', 'BWA', 'ESH']  # countries with low quality data to be ignored
    class_1_threshold = 55  # minimum number of absolute cases to be considered going into first wave
    class_1_threshold_dead = 5
    debug_death_lag = 9  # death lag for case-death ascertainment
    debug_countries_of_interest = ['USA', 'GBR', 'BRA', 'IND', 'ESP', 'FRA', 'ZAF']

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
