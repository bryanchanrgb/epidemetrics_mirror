import numpy as np
import pandas as pd

def get_restriction_start_date(government_response,country,flag):
    conversion_dict={'c1_school_closing' : 3, 'c2_workplace_closing' : 2, 'c6_stay_at_home_requirements' : 2}
    temp = government_response[(government_response['countrycode'] == country) &
                               (government_response[flag] >= conversion_dict[flag])]
    if len(temp) == 0:
        return
    return temp.iloc[np.where(temp['date'].diff() != pd.Timedelta('1 days 00:00:00'))]['date'].values

def get_restriction_end_date(government_response,country,flag):
    conversion_dict={'c1_school_closing' : 3, 'c2_workplace_closing' : 2, 'c6_stay_at_home_requirements' : 2}
    temp = government_response[(government_response['countrycode'] == country) &
                               (government_response[flag] >= conversion_dict[flag])]
    if len(temp) == 0:
        return
    return temp.iloc[np.where(temp['date'].diff() != pd.Timedelta('1 days 00:00:00'))[0] - 1]['date'].values