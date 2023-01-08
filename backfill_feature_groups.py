import pandas as pd
from functions import *
"""
LOADING HISTORICAL DATA
"""
#Collecting historical data on air quality
df_air_quality = pd.read_csv('air_quality_ny.csv')
df_air_quality.head()


df_air_quality.date = df_air_quality.date.apply(timestamp_2_time)
df_air_quality.sort_values(by = ['date'],inplace = True,ignore_index = True)

#print(df_air_quality.head())

#Weather
# df_weather = pd.read_csv('https://repo.hops.works/dev/davit/air_quality/weather.csv')

# df_weather.head(3)

# df_weather.date = df_weather.date.apply(timestamp_2_time)
# df_weather.sort_values(by=['city', 'date'],inplace=True, ignore_index=True)

# df_weather.head(3)

# print(df_weather)

"""
CONNECTING TO HOPSWORKS
"""

import hopsworks

project = hopsworks.login()

fs = project.get_feature_store() 

"""
CREATING FEATURE GROUPS
"""
#Air quality
air_quality_fg = fs.get_or_create_feature_group(
        name = 'air_quality_fg',
        description = 'Air Quality characteristics of each day',
        version = 1,
        primary_key = ['date'],
        online_enabled = True,
        event_time = 'date'
    )    

air_quality_fg.insert(df_air_quality)

air_quality_tg = air_quality_fg.create_training_data()
air_quality_tg.insert(df_air_quality_tr)

# #Weather
# weather_fg = fs.get_or_create_feature_group(
#         name = 'weather_fg',
#         description = 'Weather characteristics of each day',
#         version = 1,
#         primary_key = ['city','date'],
#         online_enabled = True,
#         event_time = 'date'
#     )    

# weather_fg.insert(df_weather)