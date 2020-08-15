#%% import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2
import pandas as pd
import os
from tqdm import tqdm
import statsmodels.api as sm
from sklearn import preprocessing

from pandas.plotting import register_matplotlib_converters 
register_matplotlib_converters()
from datetime import datetime, timedelta
import math
import os
import statistics
import statsmodels.api as sm

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix



#%% epidemiology country table

epi_cols = ["countrycode","country","date","confirmed",
            "new_per_day","new_per_day_ma","new_per_day_smooth",
            "peak_dates","peak_heights","peak_widths","peak_prominence","peak_width_heights"]
epi = epidemiology_results.loc[epidemiology_results["peak_dates"]>0,epi_cols].reset_index(drop=True)

'''
#not sure how the threshold average and max height are placed, not the same number as peaks
epi2_cols = ["threshold_average_height","threshold_max_height"]
epi2 = epidemiology_results.loc[epidemiology_results["threshold_max_height"]>0,epi2_cols].reset_index(drop=True)
epi = pd.concat([epi,epi2], axis=1, ignore_index=False)
del epi
'''

epi.to_csv("epi.csv")


#%% mobility country table
print(mobility_results.columns)

mobilities = ["transit_stations","residential","workplace",
              "parks","retail_recreation","grocery_pharmacy"]

measures = ["heights","prominences","widths"]


# 6 tables, one for each mobility
mob_transit_stations = pd.DataFrame()
mob_residential = pd.DataFrame()
mob_workplace = pd.DataFrame()
mob_parks = pd.DataFrame()
mob_retail_recreation = pd.DataFrame()
mob_grocery_pharmacy = pd.DataFrame()

mob = mobility_results.loc[(mobility_results["transit_stations_peak_dates"]>0 )| \
                   (mobility_results["residential_peak_dates"]>0) | \
                   (mobility_results["workplace_peak_dates"]>0) | \
                   (mobility_results["parks_peak_dates"]>0) | \
                   (mobility_results["retail_recreation_peak_dates"]>0) | \
                   (mobility_results["grocery_pharmacy_peak_dates"]>0),:]
      
m = "transit_stations"     
measures = ["dates","heights","prominences","widths"]
measures = [m + "_" + p for p in ["peak_" + i for i in measures]] + [m + "_" + t for t in ["trough_" + i for i in measures]]



['countrycode', 'country', 'date', 'transit_stations_smooth',
       'transit_stations', 'transit_stations_peak_dates',
       'transit_stations_trough_dates', 'transit_stations_peak_heights',
       'transit_stations_trough_heights', 'transit_stations_peak_prominences',
       'transit_stations_trough_prominences', 'transit_stations_peak_widths',
       'transit_stations_trough_widths', 'residential_smooth', 'residential',
       'residential_peak_dates', 'residential_trough_dates',
       'residential_peak_heights', 'residential_trough_heights',
       'residential_peak_prominences', 'residential_trough_prominences',
       'residential_peak_widths', 'residential_trough_widths',
       'workplace_smooth', 'workplace', 'workplace_peak_dates',
       'workplace_trough_dates', 'workplace_peak_heights',
       'workplace_trough_heights', 'workplace_peak_prominences',
       'workplace_trough_prominences', 'workplace_peak_widths',
       'workplace_trough_widths', 'parks_smooth', 'parks', 'parks_peak_dates',
       'parks_trough_dates', 'parks_peak_heights', 'parks_trough_heights',
       'parks_peak_prominences', 'parks_trough_prominences',
       'parks_peak_widths', 'parks_trough_widths', 'retail_recreation_smooth',
       'retail_recreation', 'retail_recreation_peak_dates',
       'retail_recreation_trough_dates', 'retail_recreation_peak_heights',
       'retail_recreation_trough_heights',
       'retail_recreation_peak_prominences',
       'retail_recreation_trough_prominences', 'retail_recreation_peak_widths',
       'retail_recreation_trough_widths', 'grocery_pharmacy_smooth',
       'grocery_pharmacy', 'grocery_pharmacy_peak_dates',
       'grocery_pharmacy_trough_dates', 'grocery_pharmacy_peak_heights',
       'grocery_pharmacy_trough_heights', 'grocery_pharmacy_peak_prominences',
       'grocery_pharmacy_trough_prominences', 'grocery_pharmacy_peak_widths',
       'grocery_pharmacy_trough_widths']

#%%
print(epidemiology_results.columns)



#%% lol
epi = pd.read_csv("C:/Users/bryan/OneDrive/Desktop/epi - epi.csv")

features = ["peak_heights","peak_prominence"]
X = epi.loc[:,features]
y = epi.loc[:,"genuine_peak"]

parameters = {'max_depth':[2],'min_samples_split':range(1,20),'min_samples_leaf':range(1,20)}
clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=4)
clf.fit(X=X, y=y)
model = clf.best_estimator_
print (clf.best_score_, clf.best_params_)

y_pred = model.predict(X)
pd.DataFrame(confusion_matrix(y, y_pred),
    columns=['Predicted Not Genuine', 'Predicted Genuine'],
    index=['Not Genuine', 'Genuine'])

plot_tree(model)

#%%

prominence_threshold = 5
epi["over_threshold"] = (epi["peak_prominence"]>=prominence_threshold)


pd.DataFrame(confusion_matrix(y, epi["over_threshold"]),
    columns=['Predicted Not Genuine', 'Predicted Genuine'],
    index=['Not Genuine', 'Genuine'])



#%%
os.chdir('C:/Users/bryan/OneDrive/Desktop/Epidemetrics')

#epidemiology_results = pd.read_csv("epidemiology_results_20200725.csv",index_col=0)
#mobility_results = pd.read_csv("mobility_results_20200725.csv",index_col=0)
#government_response_results = pd.read_csv("government_response_results_20200725.csv",index_col=0)

#%% mobility consolidating

# filter by prominence, arbitrary threshold
prominence_threshold = 5
mobilities = {"transit_stations":"trough",
          "retail_recreation":"trough",
          "residential":"peak"}
for m in mobilities:
    colname = m+"_"+mobilities[m]+"_prominence_over_threshold"
    mobility_results[colname] = (mobility_results[m+"_"+mobilities[m]+"_prominences"]>=prominence_threshold)

# consolidating to country level table
mob_country = pd.DataFrame()

for m in mobilities:
    measures = [m+"_"+mobilities[m]+"_heights",
                 m+"_"+mobilities[m]+"_prominences",m+"_"+mobilities[m]+"_widths"]
    for c in tqdm(mob["countrycode"].unique()):
        mob_country.loc[c,"country"]=mobility_results.loc[mobility_results["countrycode"]==c,"country"].values[0]
        temp = mobility_results.loc[(mobility_results[m+"_"+mobilities[m]+"_dates"]>0) & 
                                    (mobility_results["countrycode"]==c) &
                                    (mobility_results[m+"_"+mobilities[m]+"_prominence_over_threshold"]==True),
                                   ["date",m+"_"+mobilities[m]+"_dates"]+measures]
        number = 1
        for i in temp.index:
            # number = str(int(temp.loc[i,m+"_"+mobilities[m]+"_dates"])) # need to redefine number after filtering out false peaks
            colname = m+"_"+mobilities[m]+"_dates" + str(number)
            mob_country.loc[c,colname] = temp.loc[i,"date"]
            for p in measures:
                colname = p + "_" + str(number)
                mob_country.loc[c,colname] = temp.loc[i,p]
            number = number + 1

#%% government response consolidating

print(government_response_results.columns)

gov = government_response_results.loc[:,["countrycode","country","max_si","max_si_start_date","max_si_end_date",
                                         "max_si_end_date","max_si_currently","multiple_peaks","high_restrictions_start_date",
                                         "high_restrictions_end_date","high_restrictions_duration","high_restrictions_current",
                                         "25_duration","50_duration","75_duration"]]

measures = {"peak_heights":"max_si",
            "peak_start_date":"max_si_start_date",
            "peak_end_date":"max_si_end_date","peak_widths","peak_prominences"}
for i in government_response_results.index:
    for m in measures:
        temp = eval(government_response_results.loc[i,m])
        if isinstance(temp,set):
            for i in range(1,len((temp))):
                do stuff
                
        else:
            colname = m + "_1"


#%%

# linear trend in time after first peak/trough in mobility, controlling for SI
X1 = raw_government_response.loc[:,["stringency_index","countrycode","date"]]
X2 = mobility_results.loc[:,["residential","countrycode","date"]]
for c in X2["countrycode"].unique():
    t0 = FINAL.loc[FINAL["COUNTRYCODE"]==c,"GOV_PEAK_2_START_DATE"].values[0]
    if not isinstance(t0, datetime.date):
        t0 = FINAL.loc[FINAL["COUNTRYCODE"]==c,"GOV_MAX_SI_START_DATE"].values[0]
    if isinstance(t0, datetime.date):
        X2.loc[(X2["countrycode"]==c),"t"] = [(a-t0).days for a in X2.loc[(X2["countrycode"]==c),"date"]]


X = X1.merge(X2, on = ['countrycode','date'], how = 'left')
X.drop(columns=["date"],inplace=True)
X = X.dropna(how="any")
X = X.loc[X["t"]>=0,:]
print(X)

mob_trends = pd.DataFrame()
for c in tqdm(X["countrycode"].unique()):
    X_data = X.loc[X["countrycode"]==c,["t","stringency_index"]]
    y_data = X.loc[X["countrycode"]==c,["residential"]]
    model = sm.OLS(y_data, X_data).fit(se="HC3")
    mob_trends.loc[c,"t_coeff"] = model.params["t"]
    mob_trends.loc[c,"si_coeff"] = model.params["stringency_index"]

mob_trends.to_csv("C:/Users/bryan/OneDrive/Desktop/mob_trends.csv")

#%%

mob_trends["countrycode"]=mob_trends.index

epi = epi.merge(mob_trends, on="countrycode", how = "left")

#%%
test = epi.loc[(epi["past_first_wave"]),["entering_second_wave","in_or_past_second_wave","t_coeff"]]
test.drop_duplicates(inplace=True)
test.dropna(how="any",inplace=True)


np.mean(test.loc[(test["entering_second_wave"]) | (test["in_or_past_second_wave"]),
         "t_coeff"])
np.mean(test.loc[(test["entering_second_wave"]==False) & (test["in_or_past_second_wave"]==False),
         "t_coeff"])

#%% processing from final country table

final = pd.read_csv("C:/Users/bryan/OneDrive/Desktop/Epidemetrics/master.csv", index_col=0)

                  
#%%

# currently if there is only 1 peak, this appears as max_si and not as a peak
# fill in peak 1 values with max SI
master = final
for i in tqdm(master.index):
    if (type(master.loc[i,"GOV_PEAK_2_START_DATE"])!=datetime.date) \
        and (type(master.loc[i,"GOV_PEAK_2_START_DATE"])!=str):
        master.loc[i,"GOV_PEAK_2_HEIGHT"] = master.loc[i,"GOV_MAX_SI"]
        master.loc[i,"GOV_PEAK_2_START_DATE"] = master.loc[i,"GOV_MAX_SI_START_DATE"]
        master.loc[i,"GOV_PEAK_2_END_DATE"] = master.loc[i,"GOV_MAX_SI_END_DATE"]
        master.loc[i,"GOV_PEAK_2_WIDTH"] = master.loc[i,"GOV_MAX_SI_DURATION"]
        master.loc[i,"GOV_PEAK_2_PROMINENCES"] = master.loc[i,"GOV_MAX_SI"]

for i in range(2,6):
    measures = ["HEIGHT","START_DATE","END_DATE","WIDTH","PROMINENCES"]
    k = {"GOV_PEAK_" + str(i) + "_" + m : "GOV_PEAK_" + str(i-1) + "_" + m for m in measures}
    master.rename(columns=k, inplace=True)


#%%
    
epi_prominence_threshold = 6
mob_prominence_threshold = 5
gov_prominence_threshold = 10

master2 = master.loc[:,["COUNTRY","COUNTRYCODE","EPI_CONFIRMED"]]
epi_measures = ["DATE","VALUE","PROMINENCE","WIDTH"]
mob_measures = ["DATE","VALUE","PROMINENCE","WIDTH"]
gov_measures = ["START_DATE","END_DATE","HEIGHT","PROMINENCES","WIDTH"]

for i in tqdm(master.index):
    # epidemiology peaks
    j = 0
    for k in range(1,int(max(master["EPI_NUMBER_PEAKS"]))):
        prom = master.loc[i,"EPI_PEAK_" + str(k) + "_PROMINENCE"]
        if (not np.isnan(prom)) and (prom > epi_prominence_threshold):
            j = j + 1
            for m in epi_measures:
                colname = "EPI_PEAK_" + str(k) + "_" + m
                new_colname = "EPI_PEAK_" + str(j) + "_" + m
                master2.loc[i,new_colname] = master.loc[i,colname]
        master2.loc[i,"EPI_NUMBER_FILTERED_PEAKS"] = j
    # mobility peaks (im taking only residential)
    j = 0
    for k in range(1,4):
        prom = master.loc[i,"MOB_RESIDENTIAL_PEAK_" + str(k) + "_PROMINENCE"]
        if (not np.isnan(prom)) and (prom > mob_prominence_threshold):
            j = j + 1
            for m in mob_measures:
                colname = "MOB_RESIDENTIAL_PEAK_" + str(k) + "_" + m
                new_colname = "MOB_RESIDENTIAL_PEAK_" + str(j) + "_" + m
                master2.loc[i,new_colname] = master.loc[i,colname]
        master2.loc[i,"MOB_RESIDENTIAL_NUMBER_FILTERED_PEAKS"] = j
    # government response peaks
    j = 0
    for k in range(1,5):
        prom = master.loc[i,"GOV_PEAK_" + str(k) + "_PROMINENCES"]
        if (not np.isnan(prom)) and (prom > gov_prominence_threshold):
            j = j + 1
            for m in gov_measures:
                colname = "GOV_PEAK_" + str(k) + "_" + m
                new_colname = "GOV_PEAK_" + str(j) + "_" + m
                master2.loc[i,new_colname] = master.loc[i,colname]
        master2.loc[i,"GOV_NUMBER_FILTERED_PEAKS"] = j


#%% interactions between tables (table order epi >> gov >> mob)
#epi - gov
# difference between nth epi peak and start/end date of nth gov peak
for i in tqdm(master2.index):
    for n in range(1,int(min(master2.loc[i,"EPI_NUMBER_FILTERED_PEAKS"],master2.loc[i,"GOV_NUMBER_FILTERED_PEAKS"]))+1):
        # start date
        colname = "EPI_GOV_PEAK_" + str(n) + "_START_DATE_DIFF"
        date1 = datetime.strptime(master2.loc[i,"EPI_PEAK_" + str(n) + "_DATE"],"%Y-%m-%d")
        date2 = datetime.strptime(master2.loc[i,"GOV_PEAK_" + str(n) + "_START_DATE"],"%Y-%m-%d")
        master2.loc[i,colname] = (date1-date2).days
        # end date
        colname = "EPI_GOV_PEAK_" + str(n) + "_END_DATE_DIFF"
        date2 = datetime.strptime(master2.loc[i,"GOV_PEAK_" + str(n) + "_END_DATE"],"%Y-%m-%d")
        master2.loc[i,colname] = (date1-date2).days

#gov - mob
# difference between start date of SI peak and peak of residential mobility
for i in tqdm(master2.index):
    for n in range(1,int(min(master2.loc[i,"MOB_RESIDENTIAL_NUMBER_FILTERED_PEAKS"],master2.loc[i,"GOV_NUMBER_FILTERED_PEAKS"]))+1):
        # start date
        colname = "GOV_MOB_PEAK_" + str(n) + "_START_DATE_DIFF"
        date1 = datetime.strptime(master2.loc[i,"GOV_PEAK_" + str(n) + "_START_DATE"],"%Y-%m-%d")
        date2 = datetime.strptime(master2.loc[i,"MOB_RESIDENTIAL_PEAK_"+ str(n) + "_DATE"],"%Y-%m-%d")
        master2.loc[i,colname] = (date1-date2).days

#epi - mob
# difference between peak date of epi and peak date in mob
for i in tqdm(master2.index):
    for n in range(1,int(min(master2.loc[i,"EPI_NUMBER_FILTERED_PEAKS"],master2.loc[i,"MOB_RESIDENTIAL_NUMBER_FILTERED_PEAKS"]))+1):
        colname = "EPI_MOB_PEAK_" + str(n) + "_DATE_DIFF"
        date1 = datetime.strptime(master2.loc[i,"EPI_PEAK_"+str(n)+"_DATE"],"%Y-%m-%d")
        date2 = datetime.strptime(master2.loc[i,"MOB_RESIDENTIAL_PEAK_" + str(n) + "_DATE"],"%Y-%m-%d")
        master2.loc[i,colname] = (date1-date2).days

master2 = master2.reindex(sorted(master2.columns), axis=1)

master2.to_csv("C:/Users/bryan/OneDrive/Desktop/master2.csv")

#%% lets ploooot


        
plt.clf()
g = sns.distplot(master2["GOV_MOB_PEAK_1_START_DATE_DIFF"].dropna(how="all"))

#%%
        
plt.clf()
g = sns.distplot(master2["EPI_GOV_PEAK_2_START_DATE_DIFF"].dropna(how="all"))
print(len(master2["EPI_GOV_PEAK_2_START_DATE_DIFF"].dropna(how="all")))

plt.clf()
ycol = "EPI_GOV_PEAK_1_START_DATE_DIFF"
xcol = "EPI_GOV_PEAK_1_END_DATE_DIFF"
g = sns.scatterplot(y=ycol,x=xcol,data=master2[[xcol,ycol]].dropna(how="any"))

