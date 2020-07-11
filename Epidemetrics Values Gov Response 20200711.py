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
from scipy.optimize import curve_fit
from scipy import special
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm


#%%
os.chdir("C:/Users/bryan/OneDrive/Desktop/Epidemetrics")

#%%
# Connect to covid19db.org
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
cur.execute("SELECT source, date, country, countrycode, confirmed, dead, recovered FROM epidemiology WHERE source = 'WRD_WHOJHU' AND adm_area_1 IS NULL AND confirmed > 0;")
epidemiology = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
# new data source has recovered data! can build transmission rate now

#%% save to csv

#epidemiology.to_csv("epidemiology.csv")                         # 478,183 rows, 15 columns
government_response.to_csv("government_response.csv")           # 31,862 rows, 46 columns
#mobility.to_csv("mobility.csv")                                 # 1,040,495 rows, 17 columns

#%% get country list
cur.execute("SELECT DISTINCT country, countrycode FROM epidemiology WHERE adm_area_1 IS NULL AND confirmed IS NOT NULL;")
country_list = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])


#%% check that all dates are contiguous, surprisingly yes
for i in tqdm(epidemiology.index):
    if i == 0:
        date_temp = epidemiology.loc[i,"date"]
        country_temp = epidemiology.loc[i,"countrycode"]
    elif epidemiology.loc[i,"countrycode"] == country_temp:
        if epidemiology.loc[i,"date"] != date_temp + timedelta(days=1):
            print(country_temp + " : " + date_temp)
        date_temp = epidemiology.loc[i,"date"]
    else:
        date_temp = epidemiology.loc[i,"date"]
        country_temp = epidemiology.loc[i,"countrycode"]
# can loop through dates now
#%% calculate transmission rate
epidemiology["active"] = epidemiology["confirmed"] - epidemiology["dead"] - epidemiology["recovered"]

epidemiology = epidemiology.sort_values(by=["countrycode","date"],ascending=[True,True])
epidemiology.reset_index(inplace=True,drop=True)

# calculate t as number of days since first case
for c in tqdm(epidemiology["countrycode"].unique()):
        t0 = min(epidemiology.loc[epidemiology["countrycode"]==c,"date"])
        epidemiology.loc[epidemiology["countrycode"]==c,"t"] = [(a-t0).days for a in epidemiology.loc[epidemiology["countrycode"]==c,"date"]]

# calculate number of new cases per day
for i in tqdm(epidemiology.index):
    if i == 0:
        epidemiology.loc[i,"new_confirmed"] = np.nan
    elif epidemiology.loc[i,"countrycode"] == epidemiology.loc[i-1,"countrycode"]:
        epidemiology.loc[i,"new_confirmed"] = epidemiology.loc[i,"confirmed"] - epidemiology.loc[i-1,"confirmed"]
    else:
        epidemiology.loc[i,"new_confirmed"] = np.nan
        
# calculate transmission rate = number of new cases per (number of active cases 5 days ago)
for i in tqdm(epidemiology.index):
    if i <= 4:
        epidemiology.loc[i,"active_5d_lag"] = np.nan
        epidemiology.loc[i,"transmission"] = np.nan
    elif epidemiology.loc[i,"countrycode"] == epidemiology.loc[i-5,"countrycode"]:
        epidemiology.loc[i,"active_5d_lag"] = epidemiology.loc[i-5,"active"]
        if epidemiology.loc[i,"active_5d_lag"] >= 5:    
        # for stability of calculated transmission rate, only take data points with at least 5 active cases
            epidemiology.loc[i,"transmission"] = epidemiology.loc[i,"new_confirmed"] / epidemiology.loc[i,"active_5d_lag"]
        else:
            epidemiology.loc[i,"transmission"] = np.nan
    else:
        epidemiology.loc[i,"active_5d_lag"] = np.nan
        epidemiology.loc[i,"transmission"] = np.nan

# remove cases where transmission rate <0, caused by data quality issues with number of confirmed cases
epidemiology.loc[epidemiology["transmission"]<0,"transmission"] = np.nan

epidemiology["log_transmission"] = np.nan
epidemiology.loc[epidemiology["transmission"]>0,"log_transmission"] = np.log(epidemiology.loc[epidemiology["transmission"]>0,"transmission"])


#%% plot testing
plt.clf()
g = sns.scatterplot(x="t", y="transmission", data=epidemiology[epidemiology["transmission"]<=100])
g.set_title("Transmission Rate (Number of new cases per active cases 5 days ago) Over Time")
g.figure.savefig("transmission_time.png")

#%% plot testing
plt.clf()
g = sns.lineplot(x="t", y="log_transmission", hue = "country", data=epidemiology[epidemiology["country"].isin(["China","South Africa","Portugal","Germany","Philippines","United Kingdom"])])
g.set_title("Log Transmission Rate (Number of new cases per active cases 5 days ago) Over Time")
#g.figure.savefig("log_transmission_time.png")

#%% set up country level data
data = epidemiology.dropna(axis='index', how='any')

country_data = pd.DataFrame() 
# calculate average values for each country
for i in tqdm(data["countrycode"].unique()):
    country_data.loc[i,"countrycode"] = i
    country_data.loc[i,"country"] = data.loc[data["countrycode"]==i,"country"].values[0]
    country_data.loc[i,"avg_transmission"] = sum(data.loc[data["countrycode"]==i,"transmission"])/len(data.loc[data["countrycode"]==i,"transmission"])
    country_data.loc[i,"t0"] = min(data.loc[data["countrycode"]==i,"date"])
    country_data.loc[i,"max_t"] = max(data.loc[data["countrycode"]==i,"t"])
    country_data.loc[i,"death_rate"] = max(data.loc[data["countrycode"]==i,"dead"])/max(data.loc[data["countrycode"]==i,"confirmed"])
    country_data.loc[i,"max_confirmed"] = max(data.loc[data["countrycode"]==i,"confirmed"])

country_data["log_avg_transmission"] = 0
country_data.loc[country_data["avg_transmission"]>0,"log_avg_transmission"] = np.log(country_data.loc[country_data["avg_transmission"]>0,"avg_transmission"])

# combine with WB and WVS data 
country_statistics_wb = pd.read_csv("C:/Users/bryan/OneDrive/Desktop/Epidemetrics/Data Working 20200628/country_statistics_wb.csv")
country_statistics_wb = country_statistics_wb.drop(columns = ["Unnamed: 0","Unnamed: 0.1","source","year","country","adm_level","gid","samplesize","properties"])
country_data_wvs = country_data_wvs.drop(columns = ["Unnamed: 0","Unnamed: 0.1","source","year","country","adm_level","gid","samplesize",
                                                    "Country/region","year_start","max_t"]) #"erf_a","erf_b","erf_p","erf_p","log_erf_p_pc"

country_data = pd.merge(country_data, country_statistics_wb,  how='left', left_on=['countrycode'], right_on = ['countrycode'])
country_data = pd.merge(country_data, country_data_wvs,  how='left', left_on=['countrycode'], right_on = ['countrycode'])

country_data.to_csv("country_data_20200711.csv")

#%% plot testing
plt.clf()
g = sns.scatterplot(x="Political system: Having experts make decisions", y="avg_transmission", data=country_data)
# still not really correlated visually

#%% combine data at country-day level
gov_data = government_response.loc[government_response["stringency_index"].notnull(),
                                  ["countrycode","date","c1_school_closing",
                                   "c2_workplace_closing","c3_cancel_public_events","c4_restrictions_on_gatherings","c5_close_public_transport",
                                   "c6_stay_at_home_requirements","c7_restrictions_on_internal_movement","c8_international_travel_controls",
                                   "h1_public_information_campaigns","stringency_index"]]

data = pd.merge(data, gov_data,  how='left', left_on=['countrycode','date'], right_on = ['countrycode','date'])


#%% plot testing
plt.clf()
g = sns.scatterplot(x="stringency_index", y="log_transmission", data=data)

#%%
# transmission rate is actually the number of new cases discovered today, based on the number of active cases 5 days ago
# makes sense to also use a 5 day lag in stringency

for i in tqdm(data.index):
    if len(data.loc[(data["countrycode"]==data.loc[i,"countrycode"]) & (data["date"]==data.loc[i,"date"]-timedelta(days=5)),"countrycode"])>0:
        temp_index = data.index[(data["countrycode"]==data.loc[i,"countrycode"]) & (data["date"]==data.loc[i,"date"]-timedelta(days=5))].tolist()[0]
        data.loc[i,"stringency_index_5d_lag"]=data.loc[temp_index,"stringency_index"]
        data.loc[i,"c1_school_closing_5d_lag"]=data.loc[temp_index,"c1_school_closing"]
        data.loc[i,"c2_workplace_closing_5d_lag"]=data.loc[temp_index,"c2_workplace_closing"]
        data.loc[i,"c3_cancel_public_events_5d_lag"]=data.loc[temp_index,"c3_cancel_public_events"]
        data.loc[i,"c4_restrictions_on_gatherings_5d_lag"]=data.loc[temp_index,"c4_restrictions_on_gatherings"]
        data.loc[i,"c5_close_public_transport_5d_lag"]=data.loc[temp_index,"c5_close_public_transport"]
        data.loc[i,"c6_stay_at_home_requirements_5d_lag"]=data.loc[temp_index,"c6_stay_at_home_requirements"]
        data.loc[i,"c8_international_travel_controls_5d_lag"]=data.loc[temp_index,"c8_international_travel_controls"]
        data.loc[i,"h1_public_information_campaigns_5d_lag"]=data.loc[temp_index,"h1_public_information_campaigns"]
    else:
        data.loc[i,"stringency_index_5d_lag"]=np.nan
        data.loc[i,"c1_school_closing_5d_lag"]=np.nan
        data.loc[i,"c2_workplace_closing_5d_lag"]=np.nan
        data.loc[i,"c3_cancel_public_events_5d_lag"]=np.nan
        data.loc[i,"c4_restrictions_on_gatherings_5d_lag"]=np.nan
        data.loc[i,"c5_close_public_transport_5d_lag"]=np.nan
        data.loc[i,"c6_stay_at_home_requirements_5d_lag"]=np.nan
        data.loc[i,"c8_international_travel_controls_5d_lag"]=np.nan
        data.loc[i,"h1_public_information_campaigns_5d_lag"]=np.nan

data.dropna(subset=["stringency_index_5d_lag"], inplace = True)
data.reset_index(drop=True, inplace=True)

#%% 2 way fixed effects model, or country specific time trends (with constant)

# set up dummy variables
country_dummies = pd.get_dummies(data['country'], drop_first=True)
t_dummies = pd.get_dummies(data['t'].astype(str), drop_first=True)
country_t = country_dummies.mul(data['t'], axis=0)
col_dict = {k:k+"_t" for k in country_t.columns}
country_t.rename(columns=col_dict, inplace=True)

#%% regression
# country specific time trends model, with stringency index as x 
x_data = pd.concat([data["stringency_index_5d_lag"],country_dummies,country_t], axis=1)

model = sm.OLS(endog=data['transmission'], exog=x_data, missing="drop").fit(cov_type='HC3')
model.summary()

#%% regression with country specific time trends, adding interaction with confidence in gov
wvs_cols = ["countrycode",
            "Schwartz: It is important to this person to have a good time",
            "Schwartz: It is important to this person to always behave properly",
            "Political system: Having experts make decisions",
            "Science and technology are making our lives healthier, easier, and more comfortable",
            "Confidence: The Government"]

wvs_merge = country_data_wvs.loc[:,wvs_cols]
data = pd.merge(data, wvs_merge,  how='left', left_on=['countrycode'], right_on = ['countrycode'])

data["stringency_index_5d_lag*Confidence: The Government"] = data["stringency_index_5d_lag"]*data["Confidence: The Government"]
x_data = pd.concat([data[["stringency_index_5d_lag","Confidence: The Government","stringency_index_5d_lag*Confidence: The Government"]],country_dummies,country_t], axis=1)

model = sm.OLS(endog=data['transmission'], exog=x_data, missing="drop").fit(cov_type='HC3')
model.summary()


#%% without FE
x_data = data[["stringency_index_5d_lag","Confidence: The Government","stringency_index_5d_lag*Confidence: The Government"]]

model = sm.OLS(endog=data['transmission'], exog=x_data, missing="drop").fit(cov_type='HC3')
model.summary()