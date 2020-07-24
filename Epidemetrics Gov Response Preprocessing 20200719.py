# -*- coding: utf-8 -*-
#%% import
import psycopg2
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import math
from tqdm import tqdm
import os
import statistics
import statsmodels.api as sm


#%% working directory
#os.chdir("C:/Users/bryan/OneDrive/Desktop/Epidemetrics")

#%% Connect to covid19db.org
conn = psycopg2.connect(
    host='covid19db.org',
    port=5432,
    dbname='covid19',
    user='covid19',
    password='covid19'
)
cur = conn.cursor()

''' Tables:
epidemiology
country_statistics
government_response
mobility
weather
administrative_division
'''

#%% get government_response table
cur.execute("SELECT * FROM government_response;")
government_response = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

#%% sort table by country and date
government_response = government_response.sort_values(by=["country","date"],ascending=[True,True])
government_response.reset_index(inplace=True,drop=True)


#%% check there are no gaps in dates, yes there are no gaps
'''
for i in tqdm(government_response.index):
    if i == 0:
        date_temp = government_response.loc[i,"date"]
        country_temp = government_response.loc[i,"country"]
    elif government_response.loc[i,"country"] == country_temp:
        if government_response.loc[i,"date"] != date_temp + timedelta(days=1):
            print(country_temp + " : " + date_temp)
        date_temp = government_response.loc[i,"date"]
    else:
        date_temp = government_response.loc[i,"date"]
        country_temp = government_response.loc[i,"country"]
'''
#%% build country metrics
# set up table
country_data = pd.DataFrame()

country_data["country"] = government_response["country"].unique()

## TARUN: Thought countrycodes are unique? Are there instances of multiple countries with one countrycode?
country_data["countrycode"] = [statistics.mode(government_response.loc[government_response["country"]==c,"countrycode"]) for c in country_data["country"]]

#%% date at which SI reaches its peak level
# max level of stringency index
for c in tqdm(country_data["country"]):
    # max level of stringency index
    max_si = max(government_response.loc[government_response["country"]==c,"stringency_index"])
    country_data.loc[country_data["country"]==c,"max_si"] = max_si
    try:
        # min date at which peak level of stringency index is reached
        min_date = min(government_response.loc[(government_response["country"]==c) & (government_response["stringency_index"]==max_si),"date"])
        country_data.loc[country_data["country"]==c,"date_max_si"] = min_date
        # max date at which SI is at peak level (i.e. date at which SI started decreasing, assuming an inverse U shape)
        max_date = max(government_response.loc[(government_response["country"]==c) & (government_response["stringency_index"]==max_si),"date"])
        country_data.loc[country_data["country"]==c,"last_date_max_si"] = max_date
    except ValueError: # need to fill in nan for Gibraltar, which is all nans
        country_data.loc[country_data["country"]==c,"date_max_si"] = np.nan
        country_data.loc[country_data["country"]==c,"last_date_max_si"] = np.nan
        continue
    # flag cases where A) country is still at its max SI level, or B) country's SI has multiple peaks
    # A) currently still at max SI
    if country_data.loc[country_data["country"]==c,"last_date_max_si"].values[0] == max(government_response.loc[government_response["country"]==c,"date"]):
        country_data.loc[country_data["country"]==c,"currently_max_si"] = True
    else:
        country_data.loc[country_data["country"]==c,"currently_max_si"] = False
    # B) multiple peaks: if no multiple peaks, expect piecewise monotonicity before and after max SI
    # multiple peaks before max SI is reached:
    temp_list = government_response.loc[(government_response["country"]==c) & (government_response["date"]<=min_date),"stringency_index"]
    temp_si = ""
    counter = 0 # counts the number of continuous days of decrease/unchanged in SI (day 1 must be a decrease)
    multiple_pre = False    # whether there are multiple peaks before the max SI is reached
    for i in temp_list.index: # can rely on the fact that there are no missing dates
        if temp_si == "":
            temp_si = temp_list[i]
        elif temp_list[i] < temp_si:
            counter = counter + 1   # start counting days once there is a decrease in SI
        elif temp_list[i] == temp_si and counter > 0:
            counter = counter + 1   # count any day after a decrease where SI decreases or stays the same
        elif temp_list[i] > temp_si and counter > 0:
            counter = 0
        if counter >= 7:
            multiple_pre = True
            break
        temp_si = temp_list[i]
    country_data.loc[country_data["country"]==c,"multiple_pre"] = multiple_pre
    # multiple peaks after max SI is reached:
    multiple_post = False    # whether there are multiple peaks after the max SI is reached
    if country_data.loc[country_data["country"]==c,"currently_max_si"].values[0] == False:
        temp_list = government_response.loc[(government_response["country"]==c) & (government_response["date"]>=min_date),"stringency_index"]
        temp_si = ""
        counter = 0 # counts the number of continuous days of increase/unchanged in SI (day 1 must be a increase)
        for i in temp_list.index:
            if temp_si == "":
                temp_si = temp_list[i]
            elif temp_list[i] > temp_si:
                counter = counter + 1   #  start counting days once there is an increase in SI
            elif temp_list[i] == temp_si and counter > 0:
                counter = counter + 1   # count any day after an increase where SI increases or stays the same
            elif temp_list[i] < temp_si and counter > 0:
                counter = 0
            if counter >= 7:
                multiple_post = True
                break
            temp_si = temp_list[i]
    country_data.loc[country_data["country"]==c,"multiple_post"] = multiple_post
    country_data.loc[country_data["country"]==c,"multiple_peaks"] = country_data.loc[country_data["country"]==c,"multiple_pre"].values[0] or country_data.loc[country_data["country"]==c,"multiple_post"].values[0]


# this is not a very good definition of multiple peaks to be honest
# this doesnt really work, need to exclude small jumps
# maybe better to fit a spline onto SI as well

# duration that a country was at its max SI level
country_data["duration_max_si"] = [(a-b).days+1 for a,b in zip(country_data["last_date_max_si"],country_data["date_max_si"])]

# TARUN: I think it's probably okay to put a duration for countries at max SI.
# not sure if I should put N/A for countries that are still at their peak SI level, for these countries the duration isnt complete yet
#country_data.loc[country_data["currently_max_si"]==True,"duration_max_si"] = np.nan

#%% plotting SI
'''
for c in tqdm(country_data["country"]):
    plt.clf()
    multiple_peaks = country_data.loc[country_data["country"]==c,"multiple_peaks"].values[0]
    g = sns.lineplot(x = "date", y = "stringency_index", data = government_response.loc[government_response["country"]==c,:])
    g.set(ylabel='Stringency Index', xlabel="Date")
    g.set_title("Stringency Index Over Time for " + c + " - Multiple Peaks = " + str(multiple_peaks))
    g.figure.savefig("C:/Users/bryan/OneDrive/Desktop/Epidemetrics/SI Plots/" + str(multiple_peaks) + "_" + c + ".png")
'''

#%% first date of high restrictions
# first date at which any mandatory restrictions are implemented, i.e. any of the flags are at their max value
# c1_school_closing = 3, c2_workplace_closing = 3, c3_cancel_public_events = 2, c4_restrictions_on_gatherings = 4, c5_close_public_transport = 2, 
# c6_stay_at_home_requirements = 3, c7_restrictions_on_internal_movement = 2, c8_international_travel_controls = 4

for c in tqdm(country_data["country"]):
    try:
        min_date = min(government_response.loc[(government_response["country"]==c) & \
                                               ((government_response["c1_school_closing"]==3) | \
                                                (government_response["c2_workplace_closing"]==3) | \
                                                (government_response["c3_cancel_public_events"]==2) | \
                                                (government_response["c4_restrictions_on_gatherings"]==4) | \
                                                (government_response["c5_close_public_transport"]==2) | \
                                                (government_response["c6_stay_at_home_requirements"]==3) | \
                                                (government_response["c7_restrictions_on_internal_movement"]==2) | \
                                                (government_response["c8_international_travel_controls"]==4)) \
                                                ,"date"])
        country_data.loc[country_data["country"]==c,"date_high_restrictions"] = min_date
        max_date = max(government_response.loc[(government_response["country"]==c) & \
                                               ((government_response["c1_school_closing"]==3) | \
                                                (government_response["c2_workplace_closing"]==3) | \
                                                (government_response["c3_cancel_public_events"]==2) | \
                                                (government_response["c4_restrictions_on_gatherings"]==4) | \
                                                (government_response["c5_close_public_transport"]==2) | \
                                                (government_response["c6_stay_at_home_requirements"]==3) | \
                                                (government_response["c7_restrictions_on_internal_movement"]==2) | \
                                                (government_response["c8_international_travel_controls"]==4)) \
                                                ,"date"])
        country_data.loc[country_data["country"]==c,"last_date_high_restrictions"] = max_date
    except:
        country_data.loc[country_data["country"]==c,"date_high_restrictions"] = np.nan
        country_data.loc[country_data["country"]==c,"last_date_high_restrictions"] = np.nan
    # whether country is currently still at high level of restrictions
    if country_data.loc[country_data["country"]==c,"last_date_high_restrictions"].values[0] == max(government_response.loc[government_response["country"]==c,"date"]):
        country_data.loc[country_data["country"]==c,"currently_high_restrictions"] = True
    else:
        country_data.loc[country_data["country"]==c,"currently_high_restrictions"] = False

# duration of high restrictions
country_data["duration_high_restrictions"] = country_data["last_date_high_restrictions"]-country_data["date_high_restrictions"]
country_data["duration_high_restrictions"] = country_data["duration_high_restrictions"].dt.days
# may need to exclude those that are currently still at high restrictions

#%% splitting SI into bands: 25, 50, 75.
# first date of reaching a stringency index score of 25, 50, 75, and last date the country was at 25, 50, 75
thresholds = [25,50,75]
for i in thresholds:
    colname1 = "date_" + str(i) + "_si"
    colname2 = "last_date_" + str(i) + "_si"
    for c in tqdm(country_data["country"]):
        try:
            min_date = min(government_response.loc[(government_response["country"]==c) & (government_response["stringency_index"]>=i)
                                                    ,"date"])
            country_data.loc[country_data["country"]==c,colname1] = min_date
            max_date = max(government_response.loc[(government_response["country"]==c) & (government_response["stringency_index"]>=i)
                                                    ,"date"])
            country_data.loc[country_data["country"]==c,colname2] = max_date
        except:
            country_data.loc[country_data["country"]==c,colname1] = np.nan
            country_data.loc[country_data["country"]==c,colname2] = np.nan
    
# duration at or above 25, 50, 75
for i in thresholds:
    colname3 = "duration_" + str(i) + "_si"
    colname1 = "date_" + str(i) + "_si"
    colname2 = "last_date_" + str(i) + "_si"
    country_data[colname3] = country_data[colname2]-country_data[colname1]
    country_data[colname3] = country_data[colname3].dt.days

        
#%% optimal time lag between change in restrictions and change in mobility
# pair up restrictions and mobilility:
# c2_workplace_closing, workplace
# c4_restrictions_on_gatherings, retail_recreation
# c5_close_public_transport, transit_stations
# c6_stay_at_home_requirements, residential

#%% get mobility table

cur.execute("SELECT country, countrycode, date, transit_stations, residential, workplace, parks, retail_recreation, grocery_pharmacy FROM mobility WHERE source = 'GOOGLE_MOBILITY' AND adm_area_1 IS NULL;")
mobility = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

#%% get lagged gov response scores from a -14 day to +14 day lag

data = government_response.loc[government_response["countrycode"].isin(mobility["countrycode"].unique()),["countrycode","date","c2_workplace_closing","c4_restrictions_on_gatherings","c5_close_public_transport","c6_stay_at_home_requirements","stringency_index"]]

# lag the government response variables: positive t is backward looking, negative t is forward looking
# i.e. c2_workplace_closing_lag_-5 would be c2_workplace_closing 5 days ahead
for k in ["c2_workplace_closing","c4_restrictions_on_gatherings","c5_close_public_transport","c6_stay_at_home_requirements","stringency_index"]:
    for t in range(-14,15,1):
        colname = k + "_lag_" + str(t)
        data[colname] = data[k].shift(t)

# merge government response and mobility data tables
data = pd.merge(mobility.loc[:,["countrycode","date","workplace","retail_recreation","transit_stations","residential"]].dropna(how="all"), data, how = "left", left_on=['countrycode',"date"], right_on = ['countrycode',"date"])

#%% find the optimal lag by looping through OLS regression fits for each lag
results = pd.DataFrame()

columns = {"c2_workplace_closing":"workplace",
          "c4_restrictions_on_gatherings":"retail_recreation",
          "c5_close_public_transport":"transit_stations",
          "c6_stay_at_home_requirements":"residential",
          "stringency_index":"transit_stations"}        # actually probably better to use transit stations than residential
# residential and workplace are highly periodic, because the calculated change is based on the weekday baseline
# transit stations is a bit less periodic, and can vary more freely than residential (residential has logical upper and lower bounds just based on how much time you can/must spend at home)

for k in columns:
    column_coef = k + "_coef"
    column_se = k + "_se"
    column_rsq = k + "_rsq"
    for t in tqdm(range(-14,15,1)):
        colname = k + "_lag_" + str(t)
        model = sm.OLS(endog=data[columns[k]], exog=sm.add_constant(data[colname]), missing="drop").fit()
        results.loc[t, column_coef] = model.params[colname]     # coefficient
        results.loc[t, column_se] = model.bse[colname]      # standard error of coefficient estimate
        results.loc[t, column_rsq] = model.rsquared      # r-squared of model


#%% plot the treatment effects for each lag and the optimal lag
for k in columns:
    if columns[k] == "residential":     # for residential, the coefficient is positive so we want the max value
        opt_coef = max(results[k + "_coef"])
        opt_lag = results.loc[results[k + "_coef"]==opt_coef,:].index[0]
    else:       # for others, coefficient is negative so we want the min value
        opt_coef = min(results[k + "_coef"])
        opt_lag = results.loc[results[k + "_coef"]==opt_coef,:].index[0]
    plt.clf()
    sns.set(style="darkgrid")
    g = plt.subplots(figsize=(13,8))
    g = sns.lineplot(x = results.index, y = results[k + "_coef"])
    g.fill_between(results.index, results[k + "_coef"]+1.96*results[k + "_se"], results[k + "_coef"]-1.96*results[k + "_se"], alpha=0.2)
    plt.scatter(opt_lag, opt_coef, marker='x', s = 100, color='black')
    g.set(ylabel='OLS Coefficient', xlabel="Number of Days Lag (negative number represents future value)")
    g.set_title("Treatment Effect Against Lag Duration for the Effect of " + k + " on " + columns[k] + " Mobility")
    g.figure.savefig(k + "_optimal_lag.png")
    plt.close("all")


#%% find the optimal lag for each country
# fit an OLS regression with the indivual country's data for each lag from -14 to 14 days
for c in tqdm(country_data["countrycode"].unique()):
    temp_data = data.loc[data["countrycode"]==c,:]
    if len(temp_data) == 0:
        continue
    else:
        for k in columns:   # columns maps gov response flags to corresponding mobility measures
            temp_list = []
            for t in range(-14,15,1):
                colname = k + "_lag_" + str(t)
                # fit OLS model and record the coefficient in a list
                model = sm.OLS(endog=temp_data[columns[k]], exog=sm.add_constant(temp_data[colname]), missing="drop").fit()
                temp_list.append(model.params[colname])
            # get the lag where the value of coefficient is max/minimized as the optimal lag
            if columns[k] == "residential":
                opt_lag = temp_list.index(max(temp_list)) - 14
            else:
                opt_lag = temp_list.index(min(temp_list)) - 14
            colname = k + "_opt_lag"
            country_data.loc[country_data["countrycode"]==c,colname] = opt_lag

#%% plot testing
            
k = "stringency_index"
plt.clf()
sns.set(style="darkgrid")
g = sns.distplot(country_data[k + "_opt_lag"].dropna())


plt.clf()
sns.set(style="darkgrid")
g = sns.scatterplot(x = str(k + "_opt_lag"), y = "duration_max_si", data = country_data)

#%% plot stringency index against transit stations mobility, with the optimal lag

data["transit_stations_decrease"] = -data["transit_stations"]

for c in ["GBR"]:#country_data["countrycode"].unique():
    country = country_data.loc[country_data["countrycode"]==c,"country"].values[0]
    opt_lag = country_data.loc[country_data["countrycode"]==c,"stringency_index_opt_lag"].values[0]
    if opt_lag > 0:
        temp_label = "Stringency Index " + str(int(opt_lag)) + " Days Ago"
    elif opt_lag < 0:
        temp_label = "Stringency Index in " + str(int(-opt_lag)) + " Days"
    else:
        temp_label = ""
    plt.clf()
    sns.set(style="darkgrid")
    temp_data = data.loc[data["countrycode"]==c,["date","stringency_index","transit_stations_decrease"]].dropna(how="any")
    g = sns.lineplot(x=[a+timedelta(days=opt_lag) for a in temp_data["date"]], y=temp_data["stringency_index"], color="g", label=temp_label)
    g = sns.lineplot(x=temp_data["date"], y=temp_data["stringency_index"], color="b", label="Stringency Index")
    g = sns.lineplot(x=temp_data["date"], y=temp_data["transit_stations_decrease"], color="r", label="Mobility Decrease from Baseline (%)")
    g.set(ylabel='Stringency Index/ Decrease in Transit Stations Mobility from Baseline', xlabel="Date")
    g.set_title("Stringency Index Against Transit Stations Mobility for " + country)
    #g.figure.savefig("C:/Users/bryan/OneDrive/Desktop/Epidemetrics/SI Transit Plots" + c + "_si_transit.png")
    #plt.close("all")

    
#%%
'''
measures ideas
# date of highest SI - done
# peak SI level - done
# duration of peak SI level (flag if still at max) - need to deal with multiple peaks
# date of reaching a particular threshold of SI - done
# date of max level of any stringency flag (mandatory restrictions) - done
# date of stopping international travel
# optimal lag between change in flag and change in mobility (for each corresponding metric)

# take regional or no?

# maybe some measure of being "ahead of the curve"?
    # based on start date (e.g. first case), SI relative to global avg?
    # not really easy, may not be the same shape


'''

#%%
'''
# variable definitions:
# max_si: maximum level of stringency index (SI) a country ever reaches (global max only)
# date_max_si: date at which country first reaches its maximum level of SI (takes global max only)
# last_date_max_si: the last date at which the current SI was at its maximum level. If SI is still at its max level, this will be the current date.
# duration_si: duration in days (inclusive of beginning and end date) at which a country stayed at its global maximum of SI.
# currently_max_si: True/False flag for whether the country is current at its maximum SI level
# multiple_peaks: True/False flag for whether there are multiple peaks in the country's SI level
# date_high_restrictions: first date at which a country implements high restrictions (defined as any of the 8 restriction flags reaching its maximum possible value)
# last_date_high_restrictions: last date at which a country has any of the 8 restriction flags at its maximum level. If any are currently still at maximum, this will be the current date.
# duration_high_restrictions: duration in days (inclusive of beginning and end date) at which a country had at least one restrictions flag at its maximum level.
# date_25_si: the first date at which a country's stringency index reached a score of 25 or above. Will be nan if the country's stringency index has never been 25 or above.
# last_date_25_si: the last date at which a country's stringency index was at a score of 25 or above. If SI is currently 25 or above, this will be the current date.
# duration_25_si: duration in days (inclusive of beginning and end date) at which a country had an SI score of 25 or greater.
# similarly as above for 25, 50 and 75 SI.



'''

