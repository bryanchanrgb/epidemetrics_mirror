import os
from tqdm import tqdm
from epidemetrics import Epidemetrics
from data_provider import DataProvider
from config import Config
from epipanel import EpiPanel
from table_1 import Table1

if __name__ == '__main__':
    config = Config(os.path.dirname(os.path.realpath(__file__)))

    data_provider = DataProvider(config)
    data_provider.fetch_data(use_cache=True)
    countries = data_provider.get_countries()

    epidemetrics = Epidemetrics(config, data_provider)

    t = tqdm(countries, desc='Finding peaks for all countries')
    for country in t:
        t.set_description(f"Finding peaks for: {country}")
        t.refresh()
        try:
            epidemetrics.epi_find_peaks(country, plot=True, save=True)
        except ValueError:
            print(f'Unable to find peaks for: {country}')
        except KeyboardInterrupt:
            exit()

    epi_panel = EpiPanel(config, data_provider, epidemetrics.summary_output).get_epi_panel()

    table_1 = Table1(config, epi_panel)
    table_1.table_1()
