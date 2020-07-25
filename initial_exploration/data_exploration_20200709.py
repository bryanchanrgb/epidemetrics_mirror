# -*- coding: utf-8 -*-
#%% import
import psycopg2
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
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

#%% get epidemiology table
cur.execute("SELECT * FROM epidemiology WHERE adm_area_1 IS NULL AND confirmed IS NOT NULL;")
epidemiology = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

#restrict dataset for now to only country level, where there is data for confirmed

#%% testing get other tables

cur.execute("SELECT * FROM country_statistics WHERE source = 'WVS';")
country_statistics_wvs = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])
# looks like something's wrong with country stats EVS and WVS


#%% save to csv
'''
epidemiology.to_csv("epidemiology.csv")                         # 478,183 rows, 15 columns
government_response.to_csv("government_response.csv")           # 31,862 rows, 46 columns
country_statistics.to_csv("country_statistics.csv")             # 1048 rows, 8 columns
mobility.to_csv("mobility.csv")                                 # 1,040,495 rows, 17 columns
administrative_division.to_csv("administrative_division.csv")   #
weather.to_csv("weather.csv")                                   # 
'''

#%% get country list
cur.execute("SELECT DISTINCT country, countrycode FROM epidemiology WHERE adm_area_1 IS NULL AND confirmed IS NOT NULL;")
country_list = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

#%% process columns

epidemiology = epidemiology.loc[epidemiology["confirmed"]>0,
                                ["source","date","country","countrycode","tested","confirmed","recovered","dead"]]

epidemiology["log_confirmed"] = np.log(epidemiology["confirmed"])

epidemiology["log_dead"] = 0
epidemiology.loc[epidemiology["dead"]>0,"log_dead"] = np.log(epidemiology.loc[epidemiology["dead"]>0,"dead"])

epidemiology["t"] = 0
for c in epidemiology["country"].unique():
    if len(epidemiology.loc[epidemiology["country"]==c]) > 0:
        t0 = min(epidemiology.loc[epidemiology["country"]==c,"date"])
        if isinstance(epidemiology.loc[epidemiology["country"]==c,"date"],str):
            epidemiology.loc[epidemiology["country"]==c,"t"] = [(datetime.datetime.strptime(a,'%Y-%m-%d')-datetime.datetime.strptime(t0,'%Y-%m-%d')).days for a in epidemiology.loc[epidemiology["country"]==c,"date"]]
        else:
            epidemiology.loc[epidemiology["country"]==c,"t"] = [(a-t0).days for a in epidemiology.loc[epidemiology["country"]==c,"date"]]


epidemiology["log_t"] = np.log(epidemiology["t"]+1)

epidemiology["active"] = epidemiology["confirmed"] - epidemiology["recovered"] - epidemiology["dead"] 

epidemiology = epidemiology.sort_values(by=["country","source","t"])
epidemiology = epidemiology.reset_index(drop = True)

for i in tqdm(range(1,len(epidemiology))):
    if epidemiology.loc[i,"country"] == epidemiology.loc[i-1,"country"] and epidemiology.loc[i,"source"] == epidemiology.loc[i-1,"source"]:
        epidemiology.loc[i,"new_confirmed"] = epidemiology.loc[i,"confirmed"] - epidemiology.loc[i-1,"confirmed"]
        epidemiology.loc[i,"new_confirmed_daily"] = (epidemiology.loc[i,"confirmed"] - epidemiology.loc[i-1,"confirmed"])/(epidemiology.loc[i,"t"] - epidemiology.loc[i-1,"t"])
        epidemiology.loc[i,"new_confirmed_daily_per_active"] = epidemiology.loc[i,"new_confirmed_daily"] / epidemiology.loc[i-1,"active"]
    else:
        epidemiology.loc[i,"new_confirmed"] = np.nan
        epidemiology.loc[i,"new_confirmed_daily"] = np.nan
        epidemiology.loc[i,"new_confirmed_daily_per_active"] = np.nan

epidemiology["log_new_confirmed_daily"] = 0
epidemiology.loc[epidemiology["new_confirmed_daily"]>0,"log_new_confirmed_daily"] = np.log(epidemiology.loc[epidemiology["new_confirmed_daily"]>0,"new_confirmed_daily"])


#%% plot testing
plt.clf()
sns.set_style("whitegrid")
g = sns.lineplot(x="t", y="new_confirmed_daily", hue="country", data=epidemiology)
#g.figure.savefig("test11.png")


#%% set up functional form - curvefit number of cases
# http://www.healthdata.org/sites/default/files/files/research_articles/2020/CovidModel_Appendix.pdf
# uses an ERF functional form with parameters a, b, p
# ERF function from scipy: ERF(z) = 2/sqrt(pi)*integral(exp(-t**2), t=0..z)
# parameterized ERF: p/2(1+ERF(a(x-b))) = p/2(1+2/sqrt(pi)*integral(exp(-t**2), t=0..a(x-b)))
# Level: p controls the maximum asymptotic level that the rate can reach
# Slope: a controls the speed of the infection
# Inflection: b is the time at which the rate of change of y is maximal.
def func_erf(x, a, b, p):
    return (p/2)*(1+special.erf(a*(x-b)))

# alternatively expit function:
# expit(x) = 1/(1+exp(-x))
# p/expit(a(x-b)) = p/(1+exp(-(a(x-b))))
def func_expit(x, a, b, p):
    return p*special.expit(a*(x-b))

#%% curve fit for each country
country_list["erf_a"] = np.nan
country_list["erf_b"] = np.nan
country_list["erf_p"] = np.nan
country_list["confirmed_mse"] = np.nan
for i in tqdm(country_list["country"]):
    if len(epidemiology.loc[epidemiology["country"]==i,"country"]) > 0:
        # fit curve
        try:
            popt, pcov = curve_fit(f=func_erf, 
                                   xdata=epidemiology.loc[epidemiology["country"]==i,"t"], 
                                   ydata=epidemiology.loc[epidemiology["country"]==i,"confirmed"],
                                   p0=[0.02,50,50000],
                                   maxfev=100000)
        except RuntimeError:
            print("Fit not found for: " + i)
            continue
        # store fitted parameter estimates
        country_list.loc[country_list["country"]==i,"erf_a"] = popt[0]
        country_list.loc[country_list["country"]==i,"erf_b"] = popt[1]
        country_list.loc[country_list["country"]==i,"erf_p"] = popt[2]
        '''
        # calculate predicted values and residuals
        epidemiology.loc[epidemiology["country"]==i,"confirmed_hat"] = [func_erf(x,*popt) for x in epidemiology.loc[epidemiology["country"]==i,"t"]]
        epidemiology.loc[epidemiology["country"]==i,"confirmed_residual"] = epidemiology.loc[epidemiology["country"]==i,"confirmed"] - epidemiology.loc[epidemiology["country"]==i,"confirmed_hat"]
        country_list.loc[country_list["country"]==i,"confirmed_mse"] = sum([a**2 for a in epidemiology.loc[epidemiology["country"]==i,"confirmed_residual"]])/len(epidemiology.loc[epidemiology["country"]==i,"confirmed_residual"])
        # plot fit and save
        plt.clf()
        g=sns.lineplot(x="t", y="confirmed", data=epidemiology.loc[epidemiology["country"]==i,], label = "data")
        g=sns.lineplot(epidemiology.loc[epidemiology["country"]==i,"t"], func_erf(epidemiology.loc[epidemiology["country"]==i,"t"],*popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        g.set(xlabel='Number of Cases', ylabel='Days Since First Case')
        g.set_title(i)
        g.figure.savefig(i + ".png")
        '''

# gets a nice fit, but doesnt really return anything there the residual could be explained by other factors
# ideally instead of number of confirmed cases, want to use something not monotonic, and less dynamic over time

# interesting question: can we predict the curve fit parameters based on other variables?

# get largest value of t to work out the stage the country is by comparing the fit inflection point with max_t
for c in country_list["countrycode"]:
    country_list.loc[country_list["countrycode"]==c,"max_t"] = max(epidemiology.loc[epidemiology["countrycode"]==c,"t"])

#%% get country statistics from file
country_statistics = pd.read_csv("country_statistics.csv")

# split country statistics into 3 sources: WB, EVS, WVS
country_statistics_wb = country_statistics.loc[country_statistics["source"]=="WB",]
country_statistics_evs = country_statistics.loc[country_statistics["source"]=="EVS",]
country_statistics_wvs = country_statistics.loc[country_statistics["source"]=="WVS",]

#%% get the columns for WB country statistics

wb_cols = {} # get dictionary for column names'
for i in eval(country_statistics_wb.loc[0,"properties"]):
    wb_cols[i] = eval(country_statistics_wb.loc[0,"properties"])[i]["Indicator Name"]

# the year is not really relevant as we assume these are statics
# if the value is empty, the year will automatically be 1960

for i in tqdm(country_statistics_wb.index): # create a bunch of new columns for the values
    temp_dict = eval(country_statistics_wb.loc[i,"properties"])
    for j in wb_cols:
        if temp_dict[j]["Most Recent Value"] != "":
            country_statistics_wb.loc[i,j] = temp_dict[j]["Most Recent Value"]
        else:
            country_statistics_wb.loc[i,j] = np.nan

country_statistics_wb.to_csv("country_statistics_wb.csv") # save to csv
wb_cols = pd.DataFrame.from_dict(wb_cols, orient='index', columns=["Variable_Desc"])
wb_cols.to_csv("wb_cols.csv")

#%% get the columns for EVS country statistics

# there is something wrong with EVS, all countries have the same values     

evs_cols = {}
for i in country_statistics_evs.loc[215,"properties"]:
    try:
        evs_cols[i] = country_statistics_evs.loc[1,"properties"][i]["Label"]
    except KeyError:
        print("Label not found")
        continue

#%% create dataframe to store data dictionary for each question and options
evs_labels = pd.DataFrame()
temp_dict = country_statistics_evs.loc[215,"properties"]
for i in tqdm(temp_dict):
    try:
        evs_labels.loc[i,"Question_Code"] = i
        evs_labels.loc[i,"Label"] = temp_dict[i]["Label"]
        temp_cat = temp_dict[i]["Categories"]
        for j in temp_cat:
            colname = str(j[j.rfind("_")+1:])
            evs_labels.loc[i,colname] = temp_cat[j]
    except KeyError:
        print("Label not found")
        continue

#%% save data dictionary to csv
evs_labels.to_csv("evs_labels.csv") # save to csv

#%% calculate average values for each question
for k in tqdm(country_statistics_evs.index):
    temp_dict = country_statistics_evs.loc[k,"properties"]
    for i in temp_dict: # dictionary of all questions, i is the question code
        try:
            colname = temp_dict[i]["Label"]
            temp_freq = temp_dict[i]["Frequencies"] #frequencies of 1 question
        except KeyError:
            print("Label or frequencies not found: " + country_statistics_evs.loc[k,"country"] + ", " + i)
            continue
        avg_counter = 0
        total_counter = 0
        for j in temp_freq: # for each answer option
            if float(j[j.rfind("_")+1:]) > 0: # calculate average of only the >0 options
                avg_counter = avg_counter + float(j[j.rfind("_")+1:])*temp_freq[j]
                total_counter = total_counter + temp_freq[j]
        if total_counter > 0:
            country_statistics_evs.loc[k,colname] = avg_counter/total_counter
        else:
            country_statistics_evs.loc[k,colname] = np.nan

country_statistics_evs.drop(columns = "properties").to_csv("country_statistics_evs.csv") # save to csv

#%%

country_statistics_evs = pd.read_csv("country_statistics_evs.csv")

#%% keep only latest year result for each country

country_statistics_evs_consol = country_statistics_evs.loc[country_statistics_evs["adm_level"]==0,:]
for i in country_statistics_evs_consol.index:
    country_statistics_evs_consol.loc[i,"year_start"] = country_statistics_evs_consol.loc[i,"year"][0:4]

country_statistics_evs_consol = country_statistics_evs_consol.sort_values(by=["countrycode","year_start"],ascending=[True,False])
country_statistics_evs_consol.reset_index(inplace=True,drop=True)

for i in tqdm(country_statistics_evs_consol.index):
    if i == 0:
        country_temp = country_statistics_evs_consol.loc[i,"countrycode"]
        year_temp = int(country_statistics_evs_consol.loc[i,"year_start"])
    else:
        if country_statistics_evs_consol.loc[i,"countrycode"] == country_temp:
            if country_statistics_evs_consol.loc[i,"year_start"] != year_temp:
                # if same country as last row and different year, delete row
                country_statistics_evs_consol = country_statistics_evs_consol.drop([i])
        else: # if new country, reset
            country_temp = country_statistics_evs_consol.loc[i,"countrycode"]
            year_temp = int(country_statistics_evs_consol.loc[i,"year_start"])

country_statistics_evs_consol.to_csv("country_statistics_evs_consol.csv")

#%% consolidate values across multiple rows to the country level
# take only the latest year for each country
# not necessary, didnt realize the data already had the summed values
'''
country_statistics_evs_consol2 = pd.DataFrame()
evs_cols = country_statistics_evs_consol.columns.unique()
evs_cols = evs_cols[8:85]

j = 0
temp_dict = {k:np.nan for k in evs_cols}
for i in tqdm(country_statistics_evs_consol.index):
    if i == 0:
        country_temp = country_statistics_evs_consol.loc[i,"countrycode"]
        year_temp = int(country_statistics_evs_consol.loc[i,"year_start"])
        country_statistics_evs_consol2.loc[j,"source"] = "EVS"
        country_statistics_evs_consol2.loc[j,"country"] = country_statistics_evs_consol.loc[i,"country"]
        country_statistics_evs_consol2.loc[j,"countrycode"] = country_temp
        country_statistics_evs_consol2.loc[j,"year"] = country_statistics_evs_consol.loc[i,"year"]
        sample_temp = country_statistics_evs_consol.loc[i,"samplesize"]
        for k in temp_dict:
            temp_dict[k] = country_statistics_evs_consol.loc[i,k]
    else:
        if country_statistics_evs_consol.loc[i,"countrycode"] == country_temp:  # if same country as last row
            if int(country_statistics_evs_consol.loc[i,"year_start"]) == year_temp:     # if same country and year, sum sample size and average out values
                for k in temp_dict:
                    if temp_dict[k] == np.nan:  # if nans, do not sum. will mess up the weighting a bit
                        temp_dict[k] = country_statistics_evs_consol.loc[i,k]
                    elif country_statistics_evs_consol.loc[i,k] != np.nan:
                        temp_dict[k] = (temp_dict[k]*sample_temp + country_statistics_evs_consol.loc[i,k]*country_statistics_evs_consol.loc[i,"samplesize"])/(sample_temp*country_statistics_evs_consol.loc[i,"samplesize"])
                sample_temp = sample_temp + country_statistics_evs_consol.loc[i,"samplesize"]
            #else:  # if different year, skip row
        else: # if different country, start again
            country_statistics_evs_consol2.loc[j,"samplesize"] = sample_temp
            for k in temp_dict:
                country_statistics_evs_consol2.loc[j,k] = temp_dict[k]
                temp_dict[k] = country_statistics_evs_consol.loc[i,k]
            j = j + 1
            country_temp = country_statistics_evs_consol.loc[i,"countrycode"]
            year_temp = int(country_statistics_evs_consol.loc[i,"year_start"])
            country_statistics_evs_consol2.loc[j,"source"] = "EVS"
            country_statistics_evs_consol2.loc[j,"country"] = country_statistics_evs_consol.loc[i,"country"]
            country_statistics_evs_consol2.loc[j,"countrycode"] = country_temp
            country_statistics_evs_consol2.loc[j,"year"] = country_statistics_evs_consol.loc[i,"year"]
            sample_temp = country_statistics_evs_consol.loc[i,"samplesize"]
# finish last row
for k in temp_dict:
    country_statistics_evs_consol2.loc[j,k] = temp_dict[k]
country_statistics_evs_consol2.loc[j,"samplesize"] = sample_temp
'''
#%% get the columns for WVS country statistics

wvs_cols = {}
for i in country_statistics_wvs.loc[0,"properties"]:
    try:
        wvs_cols[i] = country_statistics_wvs.loc[1,"properties"][i]["Label"]
    except KeyError:
        print("Label not found")
        continue

#%% create dataframe to store data dictionary for each question and options
wvs_labels = pd.DataFrame()
temp_dict = country_statistics_wvs.loc[215,"properties"]
for i in tqdm(temp_dict):
    try:
        wvs_labels.loc[i,"Question_Code"] = i
        wvs_labels.loc[i,"Label"] = temp_dict[i]["Label"]
        temp_cat = temp_dict[i]["Categories"]
        for j in temp_cat:
            colname = str(j[j.rfind("_")+1:])
            wvs_labels.loc[i,colname] = temp_cat[j]
    except KeyError:
        print("Label not found")
        continue

#%% save data dictionary to csv
wvs_labels.to_csv("wvs_labels.csv") # save to csv

#%% calculate average values for each question
for k in tqdm(country_statistics_wvs.index):
    temp_dict = country_statistics_wvs.loc[k,"properties"]
    for i in temp_dict: # dictionary of all questions, i is the question code
        try:
            colname = temp_dict[i]["Label"]
            temp_freq = temp_dict[i]["Frequencies"] #frequencies of 1 question
        except KeyError:
            print("Label or frequencies not found: " + country_statistics_wvs.loc[k,"country"] + ", " + i)
            continue
        avg_counter = 0
        total_counter = 0
        for j in temp_freq: # for each answer option
            if float(j[j.rfind("_")+1:]) > 0: # calculate average of only the >0 options
                avg_counter = avg_counter + float(j[j.rfind("_")+1:])*temp_freq[j]
                total_counter = total_counter + temp_freq[j]
        if total_counter > 0:
            country_statistics_wvs.loc[k,colname] = avg_counter/total_counter
        else:
            country_statistics_wvs.loc[k,colname] = np.nan

country_statistics_wvs.drop(columns = "properties").to_csv("country_statistics_wvs.csv") # save to csv


#%%
country_statistics_wvs = pd.read_csv("country_statistics_wvs.csv")

#%% keep only latest year result for each country
country_statistics_wvs_consol = country_statistics_wvs.loc[country_statistics_wvs["adm_level"]==0,:]
for i in tqdm(country_statistics_wvs_consol.index):
    country_statistics_wvs_consol.loc[i,"year_start"] = country_statistics_wvs_consol.loc[i,"year"][0:4]

country_statistics_wvs_consol = country_statistics_wvs_consol.sort_values(by=["countrycode","year_start"],ascending=[True,False])
country_statistics_wvs_consol.reset_index(inplace=True,drop=True)

for i in tqdm(country_statistics_wvs_consol.index):
    if i == 0:
        country_temp = country_statistics_wvs_consol.loc[i,"countrycode"]
        year_temp = int(country_statistics_wvs_consol.loc[i,"year_start"])
    else:
        if country_statistics_wvs_consol.loc[i,"countrycode"] == country_temp:
            if country_statistics_wvs_consol.loc[i,"year_start"] != year_temp:
                # if same country as last row and different year, delete row
                country_statistics_wvs_consol = country_statistics_wvs_consol.drop([i])
        else: # if new country, reset
            country_temp = country_statistics_wvs_consol.loc[i,"countrycode"]
            year_temp = int(country_statistics_wvs_consol.loc[i,"year_start"])

country_statistics_wvs_consol.to_csv("country_statistics_wvs_consol.csv")


#%% check which cols are not as populated

cols_dict = {k: None for k in country_statistics_wvs_consol.columns}
for k in cols_dict:
    cols_dict[k] = 1-country_statistics_wvs_consol.isna().sum()[k]/len(country_statistics_wvs_consol.index)


#%% manually check the correlations for collinearity
corr_data = country_statistics_wvs_consol.drop(columns=["Unnamed: 0","source","year","country","countrycode","adm_level","year_start","samplesize","gid","Country/region"])
corr_matrix = corr_data.corr()


keep_cols = []      # keep only the columns with at least 75% of countries having a response
for i in corr_matrix.columns:
    if cols_dict[i] >= 0.75:
        keep_cols.append(i)
        
corr_matrix = corr_matrix.loc[keep_cols,keep_cols]

corr_matrix.to_csv("wvs_corr_matrix.csv")

#%% define regressor columns
short_list = ["Important in life: Family",
            "Important in life: Friends",
            "Active/Inactive membership of church or religious organization",
            "Active/Inactive membership of sport or recreation",
            "Schwartz: It is important to this person to have a good time",
            "Schwartz: It is important to this person to always behave properly",
            "Future changes: Greater respect for authority",
            "Political system: Having experts make decisions",
            "Science and technology are making our lives healthier, easier, and more comfortable",
            "How proud of nationality",
            "I see myself as a world citizen",
            "I see myself as member of my local community",
            "I see myself as citizen of the [country] nation",
            "Confidence: The Press"]

long_list = ["Important in life: Family",
            "Important in life: Friends",
            "Important in life: Work",
            "State of health (subjective)",
            "Active/Inactive membership of church or religious organization",
            "Active/Inactive membership of sport or recreation",
            "Most people can be trusted",
            "Satisfaction with your life",
            "Schwartz: It is important to this person living in secure surroundings",
            "Schwartz: It is important to this person to have a good time",
            "Schwartz: It is important to this person to always behave properly",
            "Satisfaction with financial situation of household",
            "Future changes: Greater respect for authority",
            "Political action: attending lawful/peaceful demonstrations",
            "Government responsibility",
            "Political system: Having experts make decisions",
            "Political system: Having a democratic political system",
            "Science and technology are making our lives healthier, easier, and more comfortable",
            "Importance of democracy",
            "How proud of nationality",
            "I see myself as a world citizen",
            "I see myself as member of my local community",
            "I see myself as citizen of the [country] nation",
            "Confidence: The Press",
            "Confidence: The Police",   # seems not very correlated
            "Confidence: The Government",
            "Confidence: The United Nations"]

short_list_y = short_list
short_list_y.extend(["erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t"])
long_list_y = long_list
long_list_y.extend(["erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t"])

#%% combine country epidemiology data with wvs

c = pd.read_csv("C:/Users/bryan/OneDrive/Desktop/Epidemetrics/Data Working 20200628/country_data_working.csv")
country_data = country_data.loc[:,["country","countrycode","erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t"]]


"erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t"

# remove all countries where the fitted inflection point is > max t. Basically only take countries where the curve is already starting to flatten.
country_data_clean = country_data.loc[country_data["erf_b"] < country_data["max_t"],:]


country_data_wvs = country_statistics_wvs_consol

for i in tqdm(country_data_wvs["countrycode"]):
    if not sum(country_data_clean["countrycode"]==i)==0:
        for j in ["erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t"]:
            country_data_wvs.loc[country_data_wvs["countrycode"]==i,j] = country_data_clean.loc[country_data_clean["countrycode"]==i,j].values

country_data_wvs.to_csv("country_data_wvs.csv")

country_data_wvs = country_data_wvs.loc[country_data_wvs["erf_p"]>0,:]
# only 69 countries remain

#%% plot testing


plt.clf()
g = sns.scatterplot(x="I see myself as member of my local community", y="log_erf_p_pc", hue="country", data=country_data_wvs)


#%% regression
country_data_clean = country_data_wvs.loc[:,short_list_y]
country_data_clean = country_data_clean.dropna(axis='index', how='any')

model = sm.OLS(endog=country_data_clean['log_erf_p_pc'], exog=sm.add_constant(country_data_clean[short_list])).fit(cov_type='HC3')
model.summary()

#%% prep WB country statistics for analysis
# read country statistics from csv
country_statistics_wb = pd.read_csv("country_statistics_wb.csv")
# define list of columns to keep for regression
keep_list = ["country","countrycode",
            "EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
            "EN.URB.LCTY",  #  Population in largest city - nonempty 0.7116 - Maybe - Process, check coverage
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
            "SH.DYN.MORT",  #  Mortality rate, under-5 (per 1,000 live births) - nonempty 0.8977 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
            "SH.PRV.SMOK",  #  Smoking prevalence, total (ages 15+) - nonempty 0.6791 - Maybe - Check coverage
            "SI.POV.DDAY",  #  Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) - nonempty 0.7628 - Maybe - check coverage
            "SM.POP.NETM",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.POP.TOTL",  #  Population, total - nonempty 1 - Yes - Base
            "SP.URB.GROW",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
            "IP.JRN.ARTC.SC",  #  Scientific and technical journal articles - nonempty 0.9116 - Yes - Process
            "IT.NET.USER.ZS",  #  Individuals using the Internet (% of population) - nonempty 0.986 - Yes - 
            "NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
            "SE.ADT.LITR.ZS",  #  Literacy rate, adult total (% of people ages 15 and above) - nonempty 0.7767 - Maybe - Check colinearity
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
            "SI.DST.FRST.20",  #  Income share held by lowest 20% - nonempty 0.7628 - Maybe - Check coverage
            "SL.EMP.VULN.ZS",  #  Vulnerable employment, total (% of total employment) (modeled ILO estimate) - nonempty 0.8651 - Yes - Check coverage
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
            "SP.RUR.TOTL.ZS",  #  Rural population (% of total population) - nonempty 0.9953 - Yes - 
            "BX.GRT.EXTA.CD.WD",  #  Grants, excluding technical cooperation (BoP, current US$) - nonempty 0.8279 - Yes - Process
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes - 
            "IS.AIR.GOOD.MT.K1",  #  Air transport, freight (million ton-km) - nonempty 0.8512 - Yes - Process
            "MS.MIL.TOTL.TF.ZS",  #  Armed forces personnel (% of total labor force) - nonempty 0.8 - Maybe - Check coverage
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 

country_data_working = country_statistics_wb.loc[:,keep_list]

for i in tqdm(country_list["countrycode"]):
    country_data_working.loc[country_data_working["countrycode"]==i,"erf_a"] = country_list.loc[country_list["countrycode"]==i,"erf_a"].values
    country_data_working.loc[country_data_working["countrycode"]==i,"erf_b"] = country_list.loc[country_list["countrycode"]==i,"erf_b"].values
    country_data_working.loc[country_data_working["countrycode"]==i,"erf_p"] = country_list.loc[country_list["countrycode"]==i,"erf_p"].values
    country_data_working.loc[country_data_working["countrycode"]==i,"max_t"] = country_list.loc[country_list["countrycode"]==i,"max_t"].values

# exclude the countries with no estimate for parameter
# population remaining is the subset of countries with both epidemiology data and WB statistics
country_data_working = country_data_working.loc[country_data_working["erf_p"]>0,:]

# define list of columns that need to be divided by total population
div_by_pop_list = ["EN.URB.LCTY",
                   "SM.POP.NETM",
                   "SP.URB.TOTL",
                   "IP.JRN.ARTC.SC",
                   "BX.GRT.EXTA.CD.WD",
                   "IS.AIR.GOOD.MT.K1"]

# scale number of cases parameter by population (pc for per capita)
country_data_working.loc[:,"erf_p_pc"] = country_data_working.loc[:,"erf_p"]/country_data_working.loc[:,"SP.POP.TOTL"]
# log transform p per capita just for scaling
country_data_working["log_erf_p_pc"] = np.log(country_data_working["erf_p_pc"])

for i in div_by_pop_list:
    country_data_working.loc[:,i] = country_data_working.loc[:,i]/country_data_working.loc[:,"SP.POP.TOTL"]
    country_data_working.rename(columns = {i:i+'_pc'}, inplace = True)

# for Domestic general government health expenditure per capita, express as ratio of total health expenditure
# ratio of how much health expenditure is public vs private
country_data_working.loc[:,"SH.XPD.GHED.PC.CD"] = country_data_working.loc[:,"SH.XPD.GHED.PC.CD"]/country_data_working.loc[:,"SH.XPD.CHEX.PP.CD"]
country_data_working.rename(columns = {"SH.XPD.GHED.PC.CD":"SH.XPD.GHED.PC.CD_ratio"}, inplace = True)

# urban population growth should be expressed relative to total population growth. Measure of urbanization.
country_data_working.loc[:,"SP.URB.GROW"] = country_data_working.loc[:,"SP.URB.GROW"]/country_data_working.loc[:,"SP.POP.GROW"]
country_data_working.rename(columns = {"SP.URB.GROW":"SP.URB.GROW_relative"}, inplace = True)

# save to csv
country_data_working.to_csv("country_data_working.csv")

#%% define list of independent variables
full_list = ["country","countrycode","erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t",
            "EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
            "EN.URB.LCTY_pc",  #  Population in largest city - nonempty 0.7116 - Maybe - Process, check coverage
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
            "SH.DYN.MORT",  #  Mortality rate, under-5 (per 1,000 live births) - nonempty 0.8977 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
            "SH.PRV.SMOK",  #  Smoking prevalence, total (ages 15+) - nonempty 0.6791 - Maybe - Check coverage
            "SI.POV.DDAY",  #  Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) - nonempty 0.7628 - Maybe, check coverage
            "SM.POP.NETM_pc",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.URB.GROW_relative",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes - 
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
            "IP.JRN.ARTC.SC_pc",  #  Scientific and technical journal articles - nonempty 0.9116 - Yes - Process
            "IT.NET.USER.ZS",  #  Individuals using the Internet (% of population) - nonempty 0.986 - Yes - 
            "NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
            "SE.ADT.LITR.ZS",  #  Literacy rate, adult total (% of people ages 15 and above) - nonempty 0.7767 - Maybe - Check colinearity
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
            "SI.DST.FRST.20",  #  Income share held by lowest 20% - nonempty 0.7628 - Maybe - Check coverage
            "SL.EMP.VULN.ZS",  #  Vulnerable employment, total (% of total employment) (modeled ILO estimate) - nonempty 0.8651 - Yes - Check coverage
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
            "SP.RUR.TOTL.ZS",  #  Rural population (% of total population) - nonempty 0.9953 - Yes - 
            "BX.GRT.EXTA.CD.WD_pc",  #  Grants, excluding technical cooperation (BoP, current US$) - nonempty 0.8279 - Yes - Process
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes - 
            "IS.AIR.GOOD.MT.K1_pc",  #  Air transport, freight (million ton-km) - nonempty 0.8512 - Yes - Process
            "MS.MIL.TOTL.TF.ZS",  #  Armed forces personnel (% of total labor force) - nonempty 0.8 - Maybe - Check coverage
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD_ratio",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 
# if use full list, only 81 countries out of 199 have full data

# define list with the most complete data to reduce imputation
partial_list = ["country","countrycode","erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t",
            "EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
#            "EN.URB.LCTY_pc",  #  Population in largest city - nonempty 0.7116 - Maybe - Process, check coverage
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
            "SH.DYN.MORT",  #  Mortality rate, under-5 (per 1,000 live births) - nonempty 0.8977 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
#            "SH.PRV.SMOK",  #  Smoking prevalence, total (ages 15+) - nonempty 0.6791 - Maybe - Check coverage
#            "SI.POV.DDAY",  #  Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population) - nonempty 0.7628 - Maybe, check coverage
            "SM.POP.NETM_pc",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.URB.GROW_relative",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes - 
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
            "IP.JRN.ARTC.SC_pc",  #  Scientific and technical journal articles - nonempty 0.9116 - Yes - Process
            "IT.NET.USER.ZS",  #  Individuals using the Internet (% of population) - nonempty 0.986 - Yes - 
            "NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
#            "SE.ADT.LITR.ZS",  #  Literacy rate, adult total (% of people ages 15 and above) - nonempty 0.7767 - Maybe - Check colinearity
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
#            "SI.DST.FRST.20",  #  Income share held by lowest 20% - nonempty 0.7628 - Maybe - Check coverage
            "SL.EMP.VULN.ZS",  #  Vulnerable employment, total (% of total employment) (modeled ILO estimate) - nonempty 0.8651 - Yes - Check coverage
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
            "SP.RUR.TOTL.ZS",  #  Rural population (% of total population) - nonempty 0.9953 - Yes - 
#            "BX.GRT.EXTA.CD.WD_pc",  #  Grants, excluding technical cooperation (BoP, current US$) - nonempty 0.8279 - Yes - Process
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            # CO2 and PM25 are actually not that collinear, can use both
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes -
#            "IS.AIR.GOOD.MT.K1_pc",  #  Air transport, freight (million ton-km) - nonempty 0.8512 - Yes - Process
#            "MS.MIL.TOTL.TF.ZS",  #  Armed forces personnel (% of total labor force) - nonempty 0.8 - Maybe - Check coverage
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD_ratio",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 

# using partial list, 162 out of 199 countries have full data
country_data_clean = country_data_working.loc[:,partial_list]
country_data_clean = country_data_clean.dropna(axis='index', how='any')

# using partial list, 162 out of 199 countries have full data
country_data_clean2 = country_data_working.loc[:,full_list]
country_data_clean2 = country_data_clean2.dropna(axis='index', how='any')

#%% removing poorly behaved fits
# theres a lot of outliers with extremely high p estimates (p exceeding total pop) because the country is at an early stage
# remove all countries where the fitted inflection point is > max t. Basically only take countries where the curve is already starting to flatten.
# kind of poor practice to be honest, seems like definitely selection bias here
# will also remove some countries where the fit may still be well behaved, but still at early stage

country_data_clean = country_data_clean.loc[country_data_clean["erf_b"] < country_data_clean["max_t"],:]
# resulting has no countries where erf_p_pc > 1. 115 countries remain out of total 215

# save to csv
country_data_clean.to_csv("country_data_clean.csv")

#%% plot testing

# distribution of erf_p after removing outliers
plt.clf()
plot = sns.distplot(country_data_clean["erf_p_pc"])
plt.clf()
plot = sns.distplot(country_data_clean["log_erf_p_pc"])
plt.clf()
plot = sns.distplot(country_data_clean["erf_a"])
plt.clf()
plot = sns.distplot(country_data_clean["erf_b"])

# checking visually the correlation of net migration, may make sense to use absolute value
plt.clf()
g = sns.scatterplot(x="SM.POP.NETM_pc", y="erf_p_pc", data=country_data_clean.loc[country_data_clean["SM.POP.NETM_pc"]<0,:])
# correlation doesnt look apparent for negative values

#%% regression of erf_p_pc on country statistics

country_data_clean = pd.read_csv("country_data_clean.csv")


partial_list = ["EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
            "SH.DYN.MORT",  #  Mortality rate, under-5 (per 1,000 live births) - nonempty 0.8977 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
            "SM.POP.NETM_pc",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.URB.GROW_relative",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes - 
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
            "IP.JRN.ARTC.SC_pc",  #  Scientific and technical journal articles - nonempty 0.9116 - Yes - Process
            "IT.NET.USER.ZS",  #  Individuals using the Internet (% of population) - nonempty 0.986 - Yes - 
            "NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
            "SL.EMP.VULN.ZS",  #  Vulnerable employment, total (% of total employment) (modeled ILO estimate) - nonempty 0.8651 - Yes - Check coverage
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
            "SP.RUR.TOTL.ZS",  #  Rural population (% of total population) - nonempty 0.9953 - Yes - 
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes -
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD_ratio",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 


model = sm.OLS(endog=country_data_clean['erf_a'], exog=sm.add_constant(country_data_clean[partial_list])).fit()
model.summary()
# multicollinearity problem

# can also try to regress with y = erf_b (time of inflexion point, i.e. rate of infection/recovery)
# or y= erf_a

#%% checking for multicollinearity

corr_data = country_data_clean.loc[:,partial_list]
corr_matrix = corr_data.corr()
corr_matrix.to_csv("corr_matrix.csv")

# narrow down list to remove collinear columns. some should not have been in there at all (totally collinear)
partial_list = ["country","countrycode","erf_a","erf_b","erf_p","erf_p_pc","log_erf_p_pc","max_t",
            "EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
#            "SH.DYN.MORT",  #  Mortality rate, under-5 (per 1,000 live births) - nonempty 0.8977 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
            "SM.POP.NETM_pc",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.URB.GROW_relative",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes - 
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
#            "IP.JRN.ARTC.SC_pc",  #  Scientific and technical journal articles - nonempty 0.9116 - Yes - Process
#            "IT.NET.USER.ZS",  #  Individuals using the Internet (% of population) - nonempty 0.986 - Yes - 
            "NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
#            "SL.EMP.VULN.ZS",  #  Vulnerable employment, total (% of total employment) (modeled ILO estimate) - nonempty 0.8651 - Yes - Check coverage
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
#            "SP.RUR.TOTL.ZS",  #  Rural population (% of total population) - nonempty 0.9953 - Yes - 
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes -
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD_ratio",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 

country_data_clean = country_data_working.loc[:,partial_list]
country_data_clean = country_data_clean.dropna(axis='index', how='any') # 165 out of 199 countries have full data
country_data_clean = country_data_clean.loc[country_data_clean["erf_b"] < country_data_clean["max_t"],:] # 118 remaining

country_data_clean.to_csv("country_data_clean.csv")

#%% run regression again with further filtered list
partial_list = ["EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
            "SM.POP.NETM_pc",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.URB.GROW_relative",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes - 
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
            "NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes -
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD_ratio",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 

# also use HC3 robust standard errors
model = sm.OLS(endog=country_data_clean['erf_p_pc'], exog=sm.add_constant(country_data_clean[partial_list])).fit(cov_type='HC3')
model.summary()

#%% try an alternative specification with logs
# take logs of some regressors because they are highly skweded and large variation in scale
country_data_clean["log_EN.POP.DNST"] = np.log(country_data_clean["EN.POP.DNST"]) # population density
country_data_clean["log_NY.GDP.PCAP.CN"] = np.log(country_data_clean["NY.GDP.PCAP.CN"]) # GDP per capita

partial_list = ["log_EN.POP.DNST",  #  Population density (people per sq. km of land area) - nonempty 0.9907 - Yes - 
            "SE.PRM.TENR",  #  Adjusted net enrollment rate, primary (% of primary school age children) - nonempty 0.9163 - Yes - 
            "SH.IMM.MEAS",  #  Immunization, measles (% of children ages 12-23 months) - nonempty 0.893 - Yes - 
            "SM.POP.NETM_pc",  #  Net migration - nonempty 0.8977 - Maybe - Process, check correlation
            "SP.POP.GROW",  #  Population growth (annual %) - nonempty 0.9953 - Yes - 
            "SP.URB.GROW_relative",  #  Urban population growth (annual %) - nonempty 0.9953 - Yes - 
            "EG.ELC.ACCS.ZS",  #  Access to electricity (% of population) - nonempty 0.9953 - Yes - 
            "log_NY.GDP.PCAP.CN",  #  GDP per capita (current LCU) - nonempty 0.9814 - Yes - 
            "SH.MED.BEDS.ZS",  #  Hospital beds (per 1,000 people) - nonempty 0.9302 - Yes - 
            "SH.STA.DIAB.ZS",  #  Diabetes prevalence (% of population ages 20 to 79) - nonempty 0.9721 - Yes - 
            "SP.DYN.LE00.IN",  #  Life expectancy at birth, total (years) - nonempty 0.9581 - Yes - 
            "EN.ATM.CO2E.KD.GD",  #  CO2 emissions (kg per 2010 US$ of GDP) - nonempty 0.9163 - Yes - 
            "EN.ATM.PM25.MC.M3",  #  PM2.5 air pollution, mean annual exposure (micrograms per cubic meter) - nonempty 0.9023 - Yes -
            "SE.ENR.PRSC.FM.ZS",  #  School enrollment, primary and secondary (gross), gender parity index (GPI) - nonempty 0.9395 - Yes - 
            "SH.XPD.CHEX.PP.CD",  #  Current health expenditure per capita, PPP (current international $) - nonempty 0.8744 - Yes - 
            "SH.XPD.GHED.PC.CD_ratio",  #  Domestic general government health expenditure per capita (current US$) - nonempty 0.8791 - Yes - Process
            "SP.URB.TOTL.IN.ZS"]  #  Urban population (% of total population) - nonempty 0.9953 - Yes - 

model = sm.OLS(endog=country_data_clean['log_erf_p_pc'], exog=sm.add_constant(country_data_clean[partial_list])).fit(cov_type='HC3')
model.summary()

#%% get predicted values for a, b, p

for i in country_data_clean["country"]:
    country_data_clean.loc[country_data_clean["country"]==i,"SP.POP.TOTL"] = country_data_working.loc[country_data_working["country"]==i,"SP.POP.TOTL"].values

# for erf_p
country_data_clean["pred_erf_p_pc"] = [math.exp(a) for a in model.predict()]
country_data_clean["pred_erf_p"] = country_data_clean["pred_erf_p_pc"] * country_data_clean["SP.POP.TOTL"] 

# for erf_a
model = sm.OLS(endog=country_data_clean['erf_a'], exog=sm.add_constant(country_data_clean[partial_list])).fit(cov_type='HC3')
country_data_clean["pred_erf_a"] = model.predict()

# for erf_b
model = sm.OLS(endog=country_data_clean['erf_b'], exog=sm.add_constant(country_data_clean[partial_list])).fit(cov_type='HC3')
country_data_clean["pred_erf_b"] = model.predict()

country_data_clean.to_csv("country_data_pred.csv")



#%% plot fits using predicted values from regression

path = ("C:/Users/bryan/OneDrive/Desktop/Epidemetrics/Country Predicted Fit Plots/")

for i in tqdm(country_data_clean["country"]):
        # plot fit and save
        a = country_data_clean.loc[country_data_clean["country"]==i,"erf_a"].values[0]
        b = country_data_clean.loc[country_data_clean["country"]==i,"erf_b"].values[0]
        p = country_data_clean.loc[country_data_clean["country"]==i,"erf_p"].values[0]
        pred_a = country_data_clean.loc[country_data_clean["country"]==i,"pred_erf_a"].values[0]
        pred_b = country_data_clean.loc[country_data_clean["country"]==i,"pred_erf_b"].values[0]
        pred_p = country_data_clean.loc[country_data_clean["country"]==i,"pred_erf_p"].values[0]
        plt.clf()
        g=sns.lineplot(x="t", y="confirmed", data=epidemiology.loc[epidemiology["country"]==i,], label = "Data")
        g=sns.lineplot(epidemiology.loc[epidemiology["country"]==i,"t"], 
                       func_erf(epidemiology.loc[epidemiology["country"]==i,"t"],a,b,p), 
                               label='Fit: a=%5.3f, b=%5.3f, p=%5.3f' % (a,b,p))
        g=sns.lineplot(epidemiology.loc[epidemiology["country"]==i,"t"], 
                       func_erf(epidemiology.loc[epidemiology["country"]==i,"t"],pred_a,pred_b,pred_p), 
                               label='Predicted: â=%5.3f, b̂=%5.3f, p̂=%5.3f' % (pred_a,pred_b,pred_p))
        g.set(xlabel='Days Since First Case', ylabel='Number of Cases')
        g.set_title(i)
        g.figure.savefig(path + i + ".png")


#%% testing
# checking visually the correlation
plt.clf()
g = sns.scatterplot(x="SP.URB.TOTL.IN.ZS", y="log_erf_p_pc", data=country_data_clean)

plt.clf()
plot = sns.distplot(country_data_clean["NY.GDP.PCAP.CN"])








