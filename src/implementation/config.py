from dataclasses import dataclass


@dataclass
class Config:
    abs_t0_threshold = 1000
    abs_prominence_threshold = 55  # minimum prominence
    abs_prominence_threshold_dead = 5  # minimum prominence for dead peak detection
    rel_t0_threshold = 0.05  # cases per rel_to_constant
    rel_prominence_threshold = 0.05  # prominence relative to rel_to_constant
    rel_prominence_threshold_dead = 0.0015  # prominence threshold for dead rel_to_constant
    rel_prominence_max_threshold = 500  # upper limit on relative prominence
    rel_prominence_max_threshold_dead = 50  # upper limit on relative prominencce
    rel_to_constant = 10000  # used as population reference for relative t0
    prominence_height_threshold = 0.7  # prominence must be above a percentage of the peak height
    t_sep_a = 21
    v_sep_b = 10  # v separation for sub algorithm B
    d_match = 35  # matching window for undetected case waves based on death waves
    exclude_countries = ['CMR', 'COG', 'GNQ', 'BWA', 'ESH']  # countries with low quality data to be ignored
    class_1_threshold = 55  # minimum number of absolute cases to be considered going into first wave
    class_1_threshold_dead = 5
    debug_death_lag = 9  # death lag for case-death ascertainment
    debug_countries_of_interest = ['USA', 'GBR', 'BRA', 'IND', 'ESP', 'FRA', 'ZAF']
