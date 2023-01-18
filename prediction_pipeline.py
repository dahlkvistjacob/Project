import sys
import time
from functions import get_air_prediction, get_air_quality_df, get_weather_df, get_weather_prediction, increment_one_day, time_2_timestamp

import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks", "joblib", "scikit-learn", "xgboost"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import hopsworks
    import joblib
    import numpy as np
    import pandas as pd


    #Get weather & air quality forecasts
    weather_data = get_weather_prediction("Beijing")
    df_weather = get_weather_df(weather_data)

    air_data = get_air_prediction("Beijing")
    df_air = get_air_quality_df(air_data)

    data = pd.merge(df_air, df_weather, how="left", left_on="date", right_on="date")
    data = data.sort_values(by=["date"], ascending=[False]).reset_index(drop=True)

    #Get model and predict using forecasts
    project = hopsworks.login()
    mr = project.get_model_registry()

    model = mr.get_model("xgboost", 4)
    model_path = model.download()
    model = joblib.load(model_path + "/model.pkl")

    predictions = model.predict(data)

    result = pd.DataFrame()
    result[["date"]] = data[["date"]]
    result["aqi"] = predictions
    next_day = increment_one_day(data["date"].max())
    result.append({"date": next_day, "aqi": np.NaN}, ignore_index=True)

    result.sort_values(by=["date"], ascending=[True], inplace=True)
    result["date"] = result["date"].apply(time_2_timestamp)

    #Since we are estimating "next day", we perform a shift and ignore the first day in the series.
    result["aqi"] = result["aqi"].shift(1)
    result = result.iloc[1:,:]

    result.to_csv("./AQI_Predictions.csv", sep=";", index=False)

    #Upload the results to Hopsworks.
    dataset_api = project.get_dataset_api()
    res = dataset_api.upload("./AQI_Predictions.csv", "Resources/data", overwrite=True)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()