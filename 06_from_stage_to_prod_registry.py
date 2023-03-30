# Databricks notebook source
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()
model_details = client.search_model_versions(filter_string = "name='telco_churn' and tags.`overall status` = 'May proceed to cut a release'")

for m in model_details: # the 1st model is the latest
    if m.current_stage == "Staging":
        model = m
        break
        
display (model)

# COMMAND ----------

# Sanity check to see if model can predict.
from databricks.feature_store import FeatureStoreClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

client = MlflowClient()
try:
    run_info = client.get_run(model.run_id)
    data_source = run_info.data.tags['db_table']
except Exception:
    data_source = 'ibm_telco_churn.churn_features' # make an assumption
    pass   

# Read from feature store
fs = FeatureStoreClient()

features = fs.read_table(data_source)
features = features.withColumn("TotalCharges", features.TotalCharges.cast('double'))

loaded_model = mlflow.pyfunc.load_model(f'models:/{model.name}/{model.version}')

df = features.toPandas().head(10)
X = df.drop(['churn', 'customerID'], axis=1)

try:
    lgb_pred = loaded_model.predict(X)
    client.set_model_version_tag(name=model.name, version=model.version, key="predicts", value=1)
    client.set_model_version_tag(name=model.name, version=model.version, key = "overall status", value = "Clear for prod use")
    client.transition_model_version_stage(model.name, model.version, "Production")
except Exception:
    print("Unable to predict on features.")
    client.set_model_version_tag(name=model.name, version=model.version, key="predicts", value=0)
    pass 

cust_predict = df
cust_predict['predictions'] = lgb_pred

accuracy=accuracy_score(df['churn'], cust_predict['predictions'])
print('Able to predict, LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))

display (cust_predict)

