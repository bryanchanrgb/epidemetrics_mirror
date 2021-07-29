import os
import shutil
from pathlib import Path

from pandas import DataFrame
import json

import wavefinder as wf

from data_provider import DataProvider
from config import Config


class Epidemetrics:
    def __init__(self, config: Config, data_provider: DataProvider):
        self.config = config
        self.data_provider = data_provider
        self.prepare_output_dirs(self.config.plot_path)
        self.summary_output = dict()

    @staticmethod
    def prepare_output_dirs(path: str):
        print(f'Preparing output directory: {path}')
        try:
            shutil.rmtree(path)
        except OSError as e:
            print(f"Error: {path}: {e.strerror}")
        Path(path).mkdir(parents=True, exist_ok=True)

    def find_peaks(self, country: str, field: str) -> wf.WaveList:

        data = self.data_provider.get_series(country=country, field=field)
        data = data[field]
        params = self.config.prominence_thresholds(field)
        params['rel_to_constant'] = self.config.rel_to_constant
        population = self.data_provider.get_population(country)
        prominence_threshold = max(params['abs_prominence_threshold'],
                                   min(params['rel_prominence_threshold'] * population / params['rel_to_constant'],
                                       params['rel_prominence_max_threshold']))
        series_name = 'Cases' if field == 'new_per_day_smooth' else 'Deaths'

        wavelist = wf.WaveList(data, series_name, self.config.t_sep_a, prominence_threshold,
                               params['prominence_height_threshold'])

        return wavelist

    def epi_find_peaks(self, country: str, plot: bool = False, save: bool = False) -> DataFrame:
        cases = self.data_provider.get_series(country=country, field='new_per_day_smooth')
        if len(cases) == 0:
            raise ValueError
        case_wavelist = self.find_peaks(country, field='new_per_day_smooth')

        deaths = self.data_provider.get_series(country=country, field='dead_per_day_smooth')
        if len(deaths) == 0:
            raise ValueError
        deaths_wavelist = self.find_peaks(country, field='dead_per_day_smooth')

        # run cross-validation (Sub Algorithm E) to find additional case waves from deaths waves
        cases_sub_e = wf.WaveCrossValidator(country).run(
            case_wavelist, deaths_wavelist, plot=plot, plot_path=self.config.plot_path)

        # compute plots
        if plot:
            wf.plot_peaks([case_wavelist, deaths_wavelist], country, self.config.plot_path, save)

        # store output of cross-validation to self.summary_output
        summary = []
        for row, peak in cases_sub_e.iterrows():
            peak_data = dict({"index": row, "location": peak.location, "date": cases.iloc[int(peak.location)].date,
                              "peak_ind": peak.peak_ind, "y_position": peak.y_position})
            summary.append(peak_data)
        self.summary_output[country] = summary

        return cases_sub_e

    def save_summary(self):
        json_data = dict({'data': []})
        for country, summary in self.summary_output.items():
            country_summary = dict({'country': country, 'numberOfWaves': len(summary), 'waves': []})
            for wave in summary:
                copied_wave = wave.copy()
                copied_wave['date'] = copied_wave['date'].strftime('%Y-%m-%d')
                country_summary['waves'].append(copied_wave)
            json_data['data'].append(country_summary)
        with open(os.path.join(self.config.plot_path, 'summary_output.json'), 'w') as f:
            json.dump(json_data, f)
