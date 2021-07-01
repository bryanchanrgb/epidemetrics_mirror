# How to run

```
docker-compose build
docker-compose run epidemetrics
```
 or

```
pip3 install -r ./requirements.txt
cd ./src
python3 ./main.py
```

# How to test

```
python -m pytest tests 
```
# Epidemetrics

The code has been redesigned and older versions of the scripts can be found
within the archive folder. Running the `generate_table.py` script will download
the data from the OxCOVID19 database and generate the CSV files need to
reproduce the figures in the paper. The CSVs should automatically be stored
within the data folder along with a time stamp of when the database was
accessed. Please note - figure_1.csv and figure_4.csv are delimited using semicolons ';', the remainder are deliminated using commas as usual. The `SAVE_PLOTS` Boolean argument can be toggled to generate the plots
needed for data labelling.

```
$ python generate_table.py
```

The figures can be reproduced by running `generate_R_plots.R`.

Alternatively, running `sh run-server.sh` at the command line will start an HTTP
server from which the `index.html` file will show the Vega-Lite visualisations.
The specifications for these figures are in the `js` directory. This makes use
of a simplified TopoJSON file I created (using `mapshaper` from GADM shapefiles)
and some simplified CSV files, essentially this just involved reading some data
into R and selecting the appropriate columns and exporting in a convenient
format.

## Downloading and computing summary table

Downloading the data and computing the summary statistics is done by running the
`generate_table.py` script. Depending on your internet connection this should
only take a couple of minutes.

```
$ python generate_table.py
```

This downloads and processes source data taken from the Oxford Covid database ...citation... and generates the following files:

- `figure_1a.csv` which contains the data used to plot Figure 1a. This includes the the number of days between the first recorded case globally and the country's T0. T0 has been defined as the date at which a country reaches at least 5 cumulative confirmed cases per 1 million population, and serves as an indicator of the country's epidemic start date. `figure_1a.csv` also contains the geometry data required to plot each country in a chloropleth.
- `figure_1b.csv` which contains the data used to plot Figure 1b. In addition to each country's geometry information, this data includes the current class of each country (one of either entering first wave, past first wave, entering second wave, past second wave, or other) as at the date of the last update.
- `figure_2a.csv` which contains the data used to plot Figure 2a. This includes the stringency index and t (days since T0) at the country-day level, and the country name, ISO code and class of each country.
- `figure_2b.csv` which contains the data used to plot Figure 2b. This includes the residential mobility and t (days since T0) at the country-day level, and the country name, ISO code and class of each country.
- `figure_2c.csv` which contains the data used to plot Figure 2c. This includes the government response time (defined as the number of days between T0 and the global peak of stringency), and total number of confirmed cases at the latest updated date normalized per 10000 population, class, country name and ISO code at the country level.
- `figure_3.csv` which contains the data used to plot Figure 3. This includes the number of new cases per day (raw and smoothed with a spline fit), number of deaths per day (raw and smoothed), number of tests per day (raw and smoothed), positive rate of testing, date, country name and ISO code at the country-day level.
- `figure_4.csv` which contains the data used to plot Figure 4. This includes the smoothed number of new cases per day, stringency index, smoothed residential mobility and t (days since T0) at the country-day level. It also includes the class, name, ISO code and the t at which stay at home restrictions were implemented, removed and implemented again for each country. The values for number of new cases per day and residential mobility for each country have been smoothed using a spline fit approximation to reduce measurement noise.
- `table_1.csv` which contains selected summary statistics aggregated for each wave state (entering first wave, past first wave, entering second wave, past second wave). It contains information on the duration and magnitude of the first and second wave, the duration of government stay at home restrictions, as well as certain country characteristics such as mean population density and GNI per capita. 

## Generating figures from processed data

`generate_R_plots.R` can be run to generate Figures 2-4. The plots will be saved in the plots folder.

### Figure 1a
Generated from `figure_1a.csv`. Figure 1a is a map of the world with each country colour coded by the number of days between the first confirmed case globally and the country's T0 (defined as the date at which the country reached at least 1000 cumulative cases). It documents the timeline of the virus' transmission across borders, providing a visual indication of which countries were hit earlier or later. 

### Figure 1b
Also generated from `figure_1b.csv`. Figure 1b is a map of the world with each country coded by its class, one of either entering first wave, past first wave, entering second wave, past second wave or other. It illustrates the stage at which each country finds itself at the current point in time.

### Figure 2a
![figure_2a](/plots/figure_2a.png)

Generated from `figure_2a.csv`. Figure 2a is a time series of the stringency index over time aggregated for countries in each of the classes: entering first wave, past first wave, entering second wave and past second wave, with the x axis plotting t, the date relative to each country's T0 to control for the temporal differences across countries in the date at which their infections began. This figure attempts to illustrate the difference in government response between countries in each class in both timing and amplitude, showing descriptively whether governments in countries that are entering first wave responded earlier or with a greater degree of restrictions as compared to countries that are entering second wave. 

### Figure 2b
![figure_2b](/plots/figure_2b.png)

Generated from `figure_2b.csv`. Figure 2b is a time series of residential mobility (relative to a baseline level defined as the median value for the corresponding day of the week during the 5-week period Jan 3 to Feb 6 2020) over time aggregated for countries in each of the classes: entering first wave, past first wave, entering second wave and past second wave. This figure attempts to illustrate the difference in public response between countries in each class in both timing and amplitude, with residential mobility serving as an indicator for whether people in certain countries stayed within their homes more.

### Figure 2c
![figure_2c](/plots/figure_2c.png)

Generated from `figure_2c.csv`. Figure 2c is a scatter plot with government response time (defined as the number of days between the country's epidemic start date the the date at which it reached its peak level of government stringency) plotted on the x axis, and the latest total number of confirmed cases (normalized per 10000 population) plotted on the y axis. Countries are colour coded on whether they are in first wave or second.  This figure illustrates the significance of government response time. It shows that countries which respond quickly and pre-emptively, often months before the country's cases reaches epidemic status, exhibit a lower number of total cases, and as of yet have not experienced a second wave.

### Figure 3
![figure_3](/plots/figure_3.png)

Generated from `figure_3.csv`. Figure 3 shows the number of new cases per day, deaths per day and tests per day over time for 3 selected countries: the United States, Belgium and Australia. It provides illustrative examples of countries experiencing a second wave in confirmed cases, and shows the relative significance of increases in testing in explaining the magnitude of the observed second wave in testing.

### Figure 4
Generated from `figure_4.csv`. Figure 4 is comprised of 3 time series subplots: epidemiology, government response and mobility. The top plot (epidemiology) shows the number of new cases per day, with the curve smoothed using a spline fit to reduce measurement noise. The center plot (government response) shows the stringency index, an index comprised of various indicators for government restrictions. The bottom plot (mobility) shows residential mobility, which is the time spent in residential areas (i.e. at home) relative to a baseline level defined at the beginning of 2020. Each of the 3 subplots shares an x axis: t, defined as the number of days from each country's epidemic start date T0. The aggregate curve for all countries in the set is plotted as the solid black line, with the curves for individual countries (the 10 countries with the highest total number of confirmed cases chosen) are plotted in colour. On each subplot, 3 key dates are indicated by a dashed black vertical line: the average date stay at home restrictions are implemented, lifted, and re-implemented, with each line annotated with the n, the number of countries for which this applies (e.g. n for restrictions re-implemented is the number of countries in the set for which restrictions have been implemented, removed and then subsequently re-implemented, and the average will be calculated on those countries).

The countries in the dataset are restricted to the subset of countries that are either entering or past the second wave, to characterize the general pattern in infection and response for those countries experiencing a second wave. The figure illustrates the similarities across countries in the how cases evolve over time in a clear first and second wave, and highlights shared patterns in government and public response for countries in this group, specifically that government restrictions and individual efforts to stay at home have not seemed to increase in response to the second wave as it did in the first, despite a strong resurgence in the number of new cases.


## Environment

To set up a virtual environment for the python packages specified in
`requirements.txt` run the following commands.

```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
