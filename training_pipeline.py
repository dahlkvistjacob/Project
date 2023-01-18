import time
import modal

LOCAL=False

if LOCAL == False:
    stub = modal.Stub()
    hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks", "scikit-learn", "xgboost"])
    @stub.function(image=hopsworks_image, timeout=1200, schedule=modal.Period(days=7), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

def g():
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from xgboost.sklearn import XGBRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    """
    CONNECTING TO HOPSWORKS
    """
    import hopsworks

    project = hopsworks.login()

    fs = project.get_feature_store()

    """
    FEATURE VIEW AND TRAINING DATASET RETRIEVAL
    """
    try:
        feature_view = fs.get_feature_view(
            name = 'air_quality',
            version = 30
        )
    except:
        air_quality_fg = fs.get_or_create_feature_group(
            name = 'air_quality_fg',
            version = 6
        )
        weather_fg = fs.get_or_create_feature_group(
            name = 'weather_fg',
            version = 6
        )
        query = air_quality_fg.select_all().join(weather_fg.select_all())
        feature_view = fs.create_feature_view(
            name = 'air_quality',
            version = 30,
            # transformation_functions = mapping_transformers,
            query = query
        )

    X = feature_view.get_training_data(1)[0]

    """
    MODELING
    """

    X = X.sort_values(by=["date"], ascending=[False]).reset_index(drop=True)
    X["aqi_next_day"] = X['aqi'].shift(1)
    X = X.iloc[1:,:]
    y = X.pop("aqi_next_day")

    gb = XGBRegressor(objective="reg:squarederror", colsample_bytree=0.5, learning_rate=0.01, max_depth=3, n_estimators=1000)

    category_cols = ['date','aqi', 'sunrise', 'sunset', 'conditions', 'description', 'icon', 'name']
    num_cols = [col for col in X.columns.values if col not in category_cols]

    cat_pipe = Pipeline(steps=[
        ("encoder", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.NaN))
    ])
    num_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    col_transformer = ColumnTransformer(transformers=[
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, [col for col in category_cols if col not in ["date", "aqi"]]),
        ("passthrough", "passthrough", ["date", "aqi"])
    ])

    model = Pipeline(steps=[
        ("prepocessor", col_transformer),
        ("model", gb)
    ])

    model.fit(X, y)

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

    joblib.dump(model, 'model.pkl')

    model = mr.sklearn.create_model(
        name="xgboost",
        description="XGBoost Regressor, with pipeline.",
        input_example=X.sample().to_numpy(),
        model_schema=model_schema,
        version=4
    )

    model.save('model.pkl')

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()