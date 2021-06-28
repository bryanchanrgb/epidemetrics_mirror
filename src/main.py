import os
from tqdm import tqdm

from epidemetrics import Epidemetrics
from data_provider import DataProvider
from implementation.config import Config
from table_1 import Table_1

if __name__ == '__main__':
    base_path = os.path.dirname(os.path.realpath(__file__))
    cache_path = os.path.abspath(os.path.join(base_path, '../cache'))
    plot_path = os.path.abspath(os.path.join(base_path, '../plots/algorithm_results'))
    data_path = os.path.abspath(os.path.join(base_path, '../data'))

    data_provider = DataProvider(cache_path=cache_path)
    data_provider.fetch_data(use_cache=True)

    config = Config()
    config.plot_path = plot_path
    epidemetrics = Epidemetrics(config, data_provider, plot_path=plot_path)

    countries = data_provider.get_countries()
    t = tqdm(countries, desc='Finding peaks for all countries')
    for country in t:
        t.set_description(f"Finding peaks for: {country}")
        t.refresh()
        try:
            epidemetrics.epi_find_peaks(country, plot=True, save=True)
        except ValueError:
            print(f'Unable to find peaks for: {country}')
        except KeyboardInterrupt:
            # User interrupt the program with ctrl+c
            exit()

    table_1 = Table_1(config=config, data_provider=data_provider, epimetrics=epidemetrics, data_path=data_path)
    table_1.table_1()
