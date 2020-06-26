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

#restrict dataset for now to only country level, where there is data for recovered and dead

#%% get country list
cur.execute("SELECT country, COUNT(*) FROM epidemiology WHERE adm_area_1 IS NULL AND confirmed IS NOT NULL GROUP BY country;")
country_list = pd.DataFrame(cur.fetchall(), columns=[desc[0] for desc in cur.description])

#%% process columns

epidemiology = epidemiology.loc[epidemiology["confirmed"]>0,
                                ["source","date","country","countrycode","tested","confirmed","recovered","dead"]]

epidemiology["log_confirmed"] = np.log(epidemiology["confirmed"])

epidemiology["log_dead"] = 0
epidemiology.loc[epidemiology["dead"]>0,"log_dead"] = np.log(epidemiology.loc[epidemiology["dead"]>0,"dead"])

epidemiology["t"] = 0
for c in country_list["country"]:
    if len(epidemiology.loc[epidemiology["country"]==c]) > 0:
        t0 = min(epidemiology.loc[epidemiology["country"]==c,"date"])
        epidemiology.loc[epidemiology["country"]==c,"t"] = [a.days for a in epidemiology.loc[epidemiology["country"]==c,"date"] - t0]

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
        # calculate predicted values and residuals
        epidemiology.loc[epidemiology["country"]==i,"confirmed_hat"] = [func_erf(x,*popt) for x in epidemiology.loc[epidemiology["country"]==i,"t"]]
        epidemiology.loc[epidemiology["country"]==i,"confirmed_residual"] = epidemiology.loc[epidemiology["country"]==i,"confirmed"] - epidemiology.loc[epidemiology["country"]==i,"confirmed_hat"]
        country_list.loc[country_list["country"]==i,"confirmed_mse"] = sum([a**2 for a in epidemiology.loc[epidemiology["country"]==i,"confirmed_residual"]])/len(epidemiology.loc[epidemiology["country"]==i,"confirmed_residual"])
        # plot fit and save
        plt.clf()
        g=sns.lineplot(x="t", y="confirmed", data=epidemiology.loc[epidemiology["country"]==i,], label = "data")
        g=sns.lineplot(epidemiology.loc[epidemiology["country"]==i,"t"], func_erf(epidemiology.loc[epidemiology["country"]==i,"t"],*popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
        g.set_title(i)
        g.figure.savefig(i + ".png")

# gets a nice fit, but doesnt really return anything there the residual could be explained by other factors
# ideally instead of number of confirmed cases, want to use something not monotonic, and less dynamic over time

# interesting question: can we predict the curve fit parameters based on other variables?


#%%

popt, pcov = curve_fit(f=func_expit, xdata=epidemiology["t"], ydata=epidemiology["confirmed"])
plt.clf()
g=sns.lineplot(x="t", y="confirmed", data=epidemiology, label = "data")
g=sns.lineplot(epidemiology["t"], func_expit(epidemiology["t"],*popt), label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))





