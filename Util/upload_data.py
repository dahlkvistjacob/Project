import hopsworks

project = hopsworks.login()
dataset_api = project.get_dataset_api()

res1 = dataset_api.upload("./Data/Air_Beijing.csv", "Resources/data", overwrite=True)
res2 = dataset_api.upload("./Data/Weather_Beijing.csv", "Resources/data", overwrite=True)
print(res1)
print(res2)
