import pandas as pd
from functions import calculate_aqi

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings("ignore")

"""
CONNECTING TO HOPSWORKS
"""
import hopsworks

project = hopsworks.login()

fs = project.get_feature_store()

"""
FEATURE VIEW AND TRAINING DATASET RETRIEVAL
"""
feature_view = fs.get_feature_view(
    name = 'air_quality',
    version = 26
)
train_data = feature_view.get_training_data(1)[0]

train_data.head()

print(train_data.head())


"""
MODELING
"""

import xgboost

train_data = train_data.sort_values(by=["date"], ascending=[False, True]).reset_index(drop=True)
train_data["aqi_next_day"] = train_data['pm25'].shift(1)
train_data["aqi_next_day"] = train_data['aqi_next_day'].apply(calculate_aqi)

train_data.head(5)

X = train_data.drop(columns=["date"]).fillna(0)
y = X.pop("aqi_next_day")

gb = xgboost.XGBRegressor()
gb.fit(X, y)

"""
MODEL VALIDATION
"""
f1_score(y.astype('int'),[int(pred) for pred in gb.predict(X)],average='micro')

y.iloc[4:10].values

pred_df = pd.DataFrame({
    'aqi_real': y.iloc[4:10].values,
    'aqi_pred': map(int, gb.predict(X.iloc[4:10]))
}
)
pred_df

"""
MODEL REGISTRY
"""

mr = project.get_model_registry()

#MODEL SCHEMA

from hsml.schema import Schema
from hsml.model_schema import ModelSchema

input_schema = Schema(X)
output_schema = Schema(y)
model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)

model_schema.to_dict()

import joblib

joblib.dump(gb, 'model.pkl')

model = mr.sklearn.create_model(
    name="xgboost",
    metrics={"f1": "0.5"},
    description="XGBoost Regressor.",
    input_example=X.sample().to_numpy(),
    model_schema=model_schema
)

model.save('model.pkl')