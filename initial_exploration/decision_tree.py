import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

from scipy.signal import find_peaks
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

DISTANCE = 21
PROMINENCE_THRESHOLD = 5
PATH = './'

warnings.filterwarnings('ignore')

LABELLED_COLUMNS = pd.read_csv(PATH + 'peak_labels.csv')
SPLINE_FITS = pd.read_csv(PATH + 'SPLINE_FITS.csv')

countries = SPLINE_FITS['countrycode'].unique()

heights = np.empty(0)
prominences = np.empty(0)
widths = np.empty(0)
distances = np.empty(0)
countrycodes = np.empty(0)

labels = np.empty(0)

for i,country in enumerate(countries):
    new_per_day_series = SPLINE_FITS[SPLINE_FITS['countrycode'] == country]['new_per_day_smooth'].values
    date_series = SPLINE_FITS[SPLINE_FITS['countrycode'] == country]['date'].values

    peak_characteristics = find_peaks(new_per_day_series, prominence=PROMINENCE_THRESHOLD, distance=DISTANCE)
    genuine_peaks = LABELLED_COLUMNS[LABELLED_COLUMNS['COUNTRYCODE'] ==
                                     country].values[0][1:4].astype(int)[0:len(peak_characteristics[0])]

    heights = np.concatenate((heights,
                                  new_per_day_series[peak_characteristics[0]]/np.max(new_per_day_series)))
    prominences = np.concatenate((prominences,
                                  peak_characteristics[1]['prominences']/np.max(new_per_day_series)))
    widths = np.concatenate((widths,
                             peak_characteristics[1]['right_bases'] - peak_characteristics[1]['left_bases']))
    distances = np.concatenate((distances, peak_characteristics[0]))
    countrycodes = np.concatenate((countrycodes,np.repeat(country,len(genuine_peaks))))
    labels = np.concatenate((labels, genuine_peaks))
    continue

X_train = np.vstack((heights, prominences, widths, distances)).T

model = DecisionTreeClassifier(max_depth=1)
model.fit(X_train, labels)
#plot_tree(model,feature_names=['heights', 'prominences', 'widths', 'distances'])

wrong = countrycodes[np.where(~(model.predict(X_train)==labels))]