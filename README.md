# Epidemetrics

## UPDATE - New Instructions

The code has been redesigned and older versions of the scripts can be found within the archive folder. Running the generate_table.py script will download the data from the OxCovid database and generate the csv files need to reproduce the figures in the paper. The CSVs should automatically be stored within the data folder along with a timestamp of when the database was accessed. The SAVE_PLOTS argument can be toggled to generate the plots needed for data labelling. 

```
$ python generate_table.py
```

The figures can be reproduced by running the following script.  

```
$ python generate_plots.py
```

## Downloading and computing summary table

Downloading the data and computing the summary statistics is done by running the
`generate_table.py` script. Depending on your internet connection this should
only take a couple of minutes.

```
$ python generate_table.py
```

This downloads and processes source data taken from the Oxford Covid database ...citation... and generates the following files:

- `FINAL.csv` which contains contains country level statistics on epidemiology, mobility and government response. The main features to note are peak characteristics (location, height, width, prominence) of the new cases per day, stringency index and each of the mobility curves, the class of each country (entering first wave, entering second wave, etc), the date of the start of the epidemic in each country T0, and the dates on which each country implements and removes stay at home restrictions. Note that this data contains all genuine and non-genuine peaks, and labels for whether each peak is genuine. The countries present in the dataset are restricted to only those for which data can be found across all of the 3 areas: epidemiology, mobility and government response.
- `epidemiology_results.csv` which contains the raw and processed data at the country-day level for epidemiology. This table contains the values of the spline fit smoothed curve, as well as the computed peak dates (peak dates are labelled n, where n is the peak number, while non-peak dates are labelled 0). The data also contains the 
- `mobility_results.csv` which contains the raw and processed data at the country-day level for mobility, including all 6 available measures from Google mobility data (residential, workplace, parks, retail_recreation, transit_stations, grocery_pharmacy). This table contains the values of the spline fit smoothed curve for each of the mobility measures, as well as the computed peak dates (peak dates are labelled n, where n is the peak number, while non-peak dates are labelled 0).
- `government_response_results.csv` which contains country level statistics on government response, notably peak characteristics for the stringency index curve, as well as the dates on which stay at home restrictions are implemented and removed.
- `fig_2_data.csv` which contains the data used to plot Figure 2. This includes the stringency index and t (days since T0) at the country-day level, and the name, ISO code and class of each country.
- `fig_3_data.csv` which contains the data used to plot Figure 3. This includes the government response time (defined as the number of days between T0 and the global peak of stringency), and total number of confirmed cases at the latest date normalized per 10000 population, class, name and ISO code at the country level.
- `fig_4_data.csv` which contains the data used to plot Figure 4. This includes the smoothed number of new cases per day, stringency index, smoothed residential mobility and t (days since T0) at the country-day level, as well as the class, name, ISO code and the t at which stay at home restrictions were implemented, removed and implemented again for each country.
- `TABLE_1.csv` which contains selected summary statistics aggregated at the class level describing key characteristics of countries entering first wave, past first wave, entering second wave and past second wave, and countries classified as other.


## Generating figures from processed data

The following scripts can be run to generate each of the plots. The plots are generated from the csv files saved by the `generate_table.py` script above.

```
$ python generate_fig_1.py
```
Generates Figure 1 from `fig_1_data.csv`. Figure 1 comprises of two plots: Figure 1a is a map of the world with each country colour coded by the number of days between 31st December 2019 and the country's T0. It documents the timeline of the virus' transmission across borders, providing a visual indication of which countries were hit earlier or later. Figure 1b is a map of the world with each country coded by its class, one of either entering first wave, past first wave, entering second wave, past second wave or other. It illustrates the stage at which each country finds itself at the current point in time.

```
$ python generate_fig_2.py
```
Generates Figure 2 from `fig_2_data.csv`. Figure 2 is a time series plot of the stringency index over time aggregated for countries in each of the classes: entering first wave, past first wave and entering second wave, with the x axis plotting t, the date relative to each country's T0 to control for the temporal differences across countries in the date at which their infections began. This figure attempts to illustrate the difference in government response between countries in each class in both timing and amplitude, showing descriptively whether governments in countries that are entering first wave responded earlier or with a greater degree of restrictions as compared to countries that are entering second wave. 

```
$ python generate_fig_3.py
```
Generates Figure 3 from `fig_3_data.csv`. Figure 3 is a scatter plot with government response time (defined as the number of days between the country's epidemic start date the the date at which it reached its peak level of government stringency) plotted on the x axis, and the latest total number of confirmed cases (normalized per 10000 population) plotted on the y axis. Countries are colour coded on whether they are in first wave or second.  This figure illustrates the significance of government response time. It shows that countries which respond quickly and pre-emptively, often months before the country's cases reaches epidemic status, exhibit a lower number of total cases, and as of yet have not experienced a second wave.

```
$ python generate_fig_4.py
```
Generates Figure 4 from `fig_4_data.csv`. Figure 4 is comprised of 3 time series: epidemiology, government response and mobility. The top plot (epidemiology) shows the number of new cases per day, with the curve smoothed using a spline fit to reduce measurement noise. The center plot (government response) shows the stringency index, an index comprised of various indicators for government restrictions. The bottom plot (mobility) shows residential mobility, which is the time spent in residential areas (i.e. at home) relative to a baseline level defined at the beginning of 2020. Each of the 3 subplots shares an x axis: t, defined as the number of days from each country's epidemic start date T0. The aggregate curve for all countries in the set is plotted as the solid black line, with the curves for individual countries (the 10 countries with the highest total number of confirmed cases chosen) are plotted in colour. On each subplot, 3 key dates are indicated by a dashed black vertical line: the average date stay at home restrictions are implemented, lifted, and re-implemented, with each line annotated with the n, the number of countries for which this applies (e.g. n for restrictions re-implemented is the number of countries in the set for which restrictions have been implemented, removed and then subsequently re-implemented, and the average will be calculated on those countries).

The countries in the dataset are restricted to the subset of countries that are either entering or past the second wave, to characterize the general pattern in infection and response for those countries experiencing a second wave. The figure illustrates the similarities across countries in the how cases evolve over time in a clear first and second wave, and highlights shared patterns in government and public response for countries in this group, specifically that government restrictions and individual efforts to stay at home have not seemed to increase in response to the second wave as it did in the first, despite a strong resurgence in the number of new cases.


## Environment

To set up a virtual environment for the python packages specified in
`requirements.txt` run the following commands.

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
