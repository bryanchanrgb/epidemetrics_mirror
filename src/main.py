from tqdm import tqdm

from processing import Epidemetrics
from data_provider import DataProvider

if __name__ == '__main__':
    data_provider = DataProvider()
    data_provider.fetch_data(use_cache=True)

    epidemetrics = Epidemetrics(data_provider)

    countries = epidemetrics.get_countries()
    t = tqdm(countries, desc='Plotting all charts')
    for country in t:
        t.set_description(f"Plotting chart for: {country}")
        t.refresh()
        try:
            epidemetrics.epi_find_peaks(country, plot=True, save=True)
        except:
            print(f'Unable to find peaks for: {country}')

    epidemetrics.table_1()
