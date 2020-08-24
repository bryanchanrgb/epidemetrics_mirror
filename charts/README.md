# Figures

The following figures are generated from the `generate_fig_n.py` scripts using data from the `fig_n_data.csv` files.

### Figure 1
Figure 1 comprises of two plots: Figure 1a is a map of the world with each country colour coded by the number of days between 31st December 2019 and the country's T0. It documents the timeline of the virus' transmission across borders, providing a visual indication of which countries were hit earlier or later. Figure 1b is a map of the world with each country coded by its class, one of either entering first wave, past first wave, entering second wave, past second wave or other. It illustrates the stage at which each country finds itself at the current point in time.

### Figure 2
Figure 2 is a time series plot of the stringency index over time aggregated for countries in each of the classes: entering first wave, past first wave and entering second wave, with the x axis plotting t, the date relative to each country's T0 to control for the temporal differences across countries in the date at which their infections began. This figure attempts to illustrate the difference in government response between countries in each class in both timing and amplitude, showing descriptively whether governments in countries that are entering first wave responded earlier or with a greater degree of restrictions as compared to countries that are entering second wave. 

### Figure 3
Figure 3 is a scatter plot with government response time (defined as the number of days between the country's epidemic start date the the date at which it reached its peak level of government stringency) plotted on the x axis, and the latest total number of confirmed cases (normalized per 10000 population) plotted on the y axis. Countries are colour coded on whether they are in first wave or second.  This figure illustrates the significance of government response time. It shows that countries which respond quickly and pre-emptively, often months before the country's cases reaches epidemic status, exhibit a lower number of total cases, and as of yet have not experienced a second wave.

### Figure 4
Figure 4 is comprised of 3 time series: epidemiology, government response and mobility. The top plot (epidemiology) shows the number of new cases per day, with the curve smoothed using a spline fit to reduce measurement noise. The center plot (government response) shows the stringency index, an index comprised of various indicators for government restrictions. The bottom plot (mobility) shows residential mobility, which is the time spent in residential areas (i.e. at home) relative to a baseline level defined at the beginning of 2020. Each of the 3 subplots shares an x axis: t, defined as the number of days from each country's epidemic start date T0. The aggregate curve for all countries in the set is plotted as the solid black line, with the curves for individual countries (the 10 countries with the highest total number of confirmed cases chosen) are plotted in colour. On each subplot, 3 key dates are indicated by a dashed black vertical line: the average date stay at home restrictions are implemented, lifted, and re-implemented, with each line annotated with the n, the number of countries for which this applies (e.g. n for restrictions re-implemented is the number of countries in the set for which restrictions have been implemented, removed and then subsequently re-implemented, and the average will be calculated on those countries).

The countries in the dataset are restricted to the subset of countries that are either entering or past the second wave, to characterize the general pattern in infection and response for those countries experiencing a second wave. The figure illustrates the similarities across countries in the how cases evolve over time in a clear first and second wave, and highlights shared patterns in government and public response for countries in this group, specifically that government restrictions and individual efforts to stay at home have not seemed to increase in response to the second wave as it did in the first, despite a strong resurgence in the number of new cases.

## Other figures currently in testing

### Figure 5
Figure 5 is a scatterplot of the duration of the first wave of cases against the average government stringency index during the time period of the first wave. It shows the relationship between harsher restrictions and the duration of the first wave.

### Figure 6
Figure 6 is a scatterplot of the integral under the stringency index curve (across the entire time period observed) against the integral under the residential mobility curve. It illustrates the strong positive correlation between stringency and mobility.

### table_figures
Plots of epidemiology, government_response and mobility time series for each country.

### Country Predicted Fit Plots
Plots for an attempt to paramaterize the curve for total number of cases by fitting an ERF function, then fitting a regression of certain country statistics (such as population statistics from the World Bank, values from the WVS) on the ERF parameters to see if much of the degree of the epidemic could be explained by measurable static differences across countries. Testing only, the fits are not good.
