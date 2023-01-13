import pandas as pd
from functions import *
import hopsworks
"""
LOADING HISTORICAL DATA
"""
WEATHER_DATA_SOURCE = ""

project = hopsworks.login()
dataset_api = project.get_dataset_api()


#Collecting historical data on air quality
air_raw_path = dataset_api.download("Resources/data/Air_Beijing.csv", None, overwrite = True)
df_air_quality = pd.read_csv(air_raw_path)
df_air_quality.head()


df_air_quality.date = df_air_quality.date.apply(timestamp_2_time)
df_air_quality.sort_values(by = ['date'], inplace = True, ignore_index = True)

# print(df_air_quality.head())

# Weather
weather_raw_path = dataset_api.download("Resources/data/Weather_Beijing.csv", None, overwrite = True)
df_weather = pd.read_csv(weather_raw_path)

df_weather.head(3)

df_weather.datetime = df_weather.datetime.apply(timestamp_2_time_hyphen)
df_weather = df_weather.rename(columns={"datetime": "date"})
df_weather = df_weather.drop(columns=["preciptype", "stations", "sealevelpressure"])
# df_weather["date"] = df_weather["datetime"]
print(df_weather.head(3))
df_weather.sort_values(by=['date'], inplace=True, ignore_index=True)

df_weather.head(3)

print(df_weather)

"""
CONNECTING TO HOPSWORKS
"""


fs = project.get_feature_store()

"""
CREATING FEATURE GROUPS

"""

#Weather
weather_fg = fs.get_or_create_feature_group(
        name = 'weather_fg',
        description = 'Weather characteristics of each day',
        version = 4,
        primary_key = ['date'],
        online_enabled = True,
        event_time = 'date'
    )


#Air quality
air_quality_fg = fs.get_or_create_feature_group(
        name = 'air_quality_fg',
        description = 'Air Quality characteristics of each day',
        version = 4,
        primary_key = ['date'],
        online_enabled = True,
        event_time = 'date'
    )

air_quality_fg.insert(df_air_quality)

#air_quality_tg = air_quality_fg.create_training_data()
#air_quality_tg.insert(df_air_quality_tr)

weather_fg.insert(df_weather)