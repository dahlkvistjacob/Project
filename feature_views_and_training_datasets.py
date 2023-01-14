
"""
CONNECTING TO HOPSWORKS
"""

import hopsworks

project = hopsworks.login()

fs = project.get_feature_store()

air_quality_fg = fs.get_or_create_feature_group(
    name = 'air_quality_fg',
    version = 5
)
weather_fg = fs.get_or_create_feature_group(
    name = 'weather_fg',
    version = 5
)

query = air_quality_fg.select_all().join(weather_fg.select_all())

query.read()

"""
FEATURE VIEW CREATION AND RETRIEVING
"""
query = air_quality_fg.select_all().join(weather_fg.select_all())

query_show = query.show(5)
col_names = query_show.columns

query_show

# Transformation functions

[t_func.name for t_func in fs.get_transformation_functions()]

category_cols = ['date','aqi', 'sunrise', 'sunset', 'conditions', 'description', 'icon', 'name']

mapping_transformers = {col_name:fs.get_transformation_function(name='standard_scaler') for col_name in col_names if col_name not in category_cols}
category_cols = {col_name:fs.get_transformation_function(name='label_encoder') for col_name in category_cols if col_name not in ['date','aqi']}

mapping_transformers.update(category_cols)

feature_view = fs.create_feature_view(
    name = 'air_quality',
    version = 26,
    transformation_functions = mapping_transformers,
    query = query
)

td = feature_view.create_training_data()
