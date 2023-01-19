<h1>Project</h1>

<h2>Purpose</h2>

The purpose of the project is to build a scalable Prediction service using a real world data source. The source is supposed to be updated regularly and the results should include air quality prediction for the following 7 days, for a specific city. The predictions should then be communicated via an UI.

<h2>Data</h2>

Historical data and APIs have been collected from both https://aqicn.org/api/ (air quality) and https://www.visualcrossing.com/ (weather). The city chosen was Beijing.
The data collected contains air quality related features such as pm25 and o3, and weather related features such as temperature and humidity. Both data sources comes with a date feature.

<h2>Libraries</h2>

- Pandas
- Modal
- Hopsworks
- Gradio
- XGBoost
- Joblib
- Numpy
- Datetime

<h2>Project Architecture</h2>

This project is divided into 4 parts:

1. Backfilling of features to feature groups
2. Feature pipeline
3. Training and feature view pipeline
4. Prediction pipeline

<h2>Implementation</h2>

1. The first module loads historical data about weather and air quality for Beijing. Some columns get dropped. It then connects to Hopsworks and creates a feature group for both weather and air quality. 
2. The second module parses the data and passes it to the feature groups created in the last step.
3. In the training pipeline a feature view is added to hopsworks with the the weather and air quality data combined. It then fetches training data, which is used to train an XGBoost Regressor.It then retrieves the model registry and creates a model schema.
4. Lastly, the prediction pipeline collects forecasts of weather and air quality. The model is then retrieved and predicts air quality by using the earlier collected forecasts. The results are uploaded to hopsworks in a csv file.

Worth noting is that the hyperparameter has been tuned for a more optimized model.
The results are later showcased at HuggingSpace using Gradio, as a graph for the following 7 days.


<h2>Results</h2>

<img src="content/feature_importance.png"
     alt="Features"
     style="float: under; margin-under: 5px;" />

Figure 1. Feature Importance

<img src="content/prediction_on_test_set.png"
     alt="Features"
     style="float: under; margin-under: 5px;" />

Figure 2. Prediction vs test set

Figure 1 displays the importance of the various features. It can be concluded that the air quality features such as pm25 and o3 is very significant. However, many weather related features are important as well, humidity for example. Figure 2 displays the predicted air quality compared to the test values. It is strongly correlated as can be seen in the figure. However, it is important to keep in mind that the model might be a bit overfitted. Chi-Squared fit was calculated to test the goodness of fit of an observed data point to the predicted, and the results were a value of 0.40.

<img src="content/huggingface.jpg"
     alt="Features"
     style="float: under; margin-under: 5px;" />
Figure 3. Hugging Face interface

Hugging Face URL: https://huggingface.co/spaces/tlord/air_quality_index

