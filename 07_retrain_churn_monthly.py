# Databricks notebook source
# MAGIC %md
# MAGIC ## Monthly Model Retrain
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step7.png?raw=true">

# COMMAND ----------

# DBTITLE 1,Load Features from latest data
# When you create the FeatureStoreClient, specify the remote workspaces with the arguments feature_store_uri and model_registry_uri.
from databricks.feature_store import FeatureStoreClient

# Due to complex configuration required, we will read from the local rather then remote feature store for the time being
# fs = FeatureStoreClient(feature_store_uri=feature_store_uri, model_registry_uri=model_registry_uri)
fs = FeatureStoreClient()
customer_features_df_raw = fs.read_table(
  name='ibm_telco_churn.churn_features',
)
customer_features_df = customer_features_df_raw.withColumn("TotalCharges", customer_features_df_raw.TotalCharges.cast('double'))

display(customer_features_df)

# COMMAND ----------

# DBTITLE 1,Train
import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import lightgbm as lgb

mlflow.autolog()
mlflow.sklearn.autolog(log_input_examples=True, silent=True)
mlflow.end_run()

curr_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment("/Users/" + curr_user + "/02a_train_log_model_lgbm")

with mlflow.start_run(run_name='lgbm predict churn') as run:
    # Extract features & labels
    training_df=customer_features_df.toPandas()
    X = training_df.drop(['churn', 'customerID'], axis=1)
    y = training_df.churn

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    lgb_params = {
        "colsample_bytree": 0.44577278777254614,
        "lambda_l1": 0.2266629157748783,
        "lambda_l2": 10.893970881329095,
        "learning_rate": 1.2449551085899786,
        "max_bin": 335,
        "max_depth": 6,
        "min_child_samples": 107,
        "n_estimators": 15,
        "num_leaves": 54,
        "path_smooth": 72.81495313163249,
        "subsample": 0.6725811904726468,
        "random_state": 685970732,
    }
    

    lgbm_clf  = lgb.LGBMClassifier(**lgb_params)
    lgbm_clf.fit(X_train,y_train)
    
    target_col = "churn"
    # Log metrics for the training set
    mlflow_model = mlflow.models.Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=lgbm_clf)
    X_train[target_col] = y_train
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train,
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": 1 }
    )
    display(training_eval_result.metrics)
    
    # Log metrics for the test set
    X_test[target_col] = y_test
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test,
        targets=target_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": 1 }
    )
    display(test_eval_result.metrics)
    
    mlflow.set_tag(key='db_table', value='ibm_telco_churn.churn_features')
    mlflow.set_tag(key='demographic_vars', value='SeniorCitizen,gender_Female')
    
    # in lieu of this use the mlfow eval function to generate metrics
    X_test = X_test.drop(['churn'], axis=1)
    lgb_pred = lgbm_clf.predict(X_test)

    accuracy=accuracy_score(lgb_pred, y_test)
    
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, lgb_pred)))
    
    # If this is notebook is ran as a task in a job, make the run_id available
    # to downstream tasks
    dbutils.jobs.taskValues.set(key = 'run_id', value = run.info.run_id)


# COMMAND ----------

# DBTITLE 1,Register the New Model
import mlflow
from mlflow.tracking import MlflowClient

run_id = run.info.run_id
client = MlflowClient()

model_name = "telco_churn"
model_uri = f"runs:/{run_id}/model"
model_details = mlflow.register_model(model_uri, model_name)

client.set_tag(run_id, key='db_table', value='ibm_telco_churn.churn_features')
client.set_tag(run_id, key='demographic_vars', value='SeniorCitizen,gender_Female')

# COMMAND ----------

# DBTITLE 1,Add Descriptions
client.update_registered_model(
  name=model_details.name,
  description="This model predicts whether a customer will churn using features from the ibm_telco_churn database.  It is used to update the Telco Churn Dashboard in DB SQL."
)

client.update_model_version(
  name=model_details.name,
  version=model_details.version,
  description="This model version was built using XGBoost. Eating too much cake is the sin of gluttony. However, eating too much pie is okay because the sin of pie is always zero."
)

# COMMAND ----------

# DBTITLE 1,Transition to Staging
# This is done to demo the use of REST API, this can also be done using Python alhough there may not
# be 1-1 exact match of functionality
# Helper function
import mlflow
from mlflow.utils.rest_utils import http_request
import json

def client():
  return mlflow.tracking.client.MlflowClient()

host_creds = client()._tracking_client.store.get_host_creds()
host = host_creds.host
token = host_creds.token

def mlflow_call_endpoint(endpoint, method, body='{}'):
  if method == 'GET':
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, params=json.loads(body))
  else:
      response = http_request(
          host_creds=host_creds, endpoint="/api/2.0/mlflow/{}".format(endpoint), method=method, json=json.loads(body))
  return response.json()


# COMMAND ----------

# Transition request to staging
staging_request = {'name': model_name, 'version': model_details.version, 'stage': 'Staging', 'archive_existing_versions': 'true'}
mlflow_call_endpoint('transition-requests/create', 'POST', json.dumps(staging_request))
mlflow_call_endpoint('transition-requests/approve', 'POST', json.dumps(staging_request))

# COMMAND ----------

# Leave a comment for the ML engineer who will be reviewing the tests, NEXT PROCEED TO VALIDATE AND THEN MOVE TO PROD.
comment = "This is a new model from the periodic retrain job."
comment_body = {'name': model_name, 'version': model_details.version, 'comment': comment}
mlflow_call_endpoint('comments/create', 'POST', json.dumps(comment_body))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity checks
# MAGIC ## Start with prediction

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

client = MlflowClient()
try:
    run_info = client.get_run(run_id)
    data_source = run_info.data.tags['db_table']
except Exception:
    data_source = 'ibm_telco_churn.churn_features' # make an assumption
    pass   

# Read from feature store
fs = FeatureStoreClient()

features = fs.read_table(data_source)
features = features.withColumn("TotalCharges", features.TotalCharges.cast('double'))

loaded_model = mlflow.pyfunc.load_model(f'models:/{model_name}/{model_details.version}')

df = features.toPandas()
X = df.drop(['churn', 'customerID'], axis=1)

try:
    lgb_pred = loaded_model.predict(X)
    client.set_model_version_tag(name=model_name, version=model_details.version, key="predicts", value=1)
except Exception:
    print("Unable to predict on features.")
    client.set_model_version_tag(name=model_name, version=model_details.version, key="predicts", value=0)
    pass 

cust_predict = df
cust_predict['predictions'] = lgb_pred

accuracy=accuracy_score(df['churn'], cust_predict['predictions'])
print('Able to predict, LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))

display (cust_predict)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC 
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC 
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

client = MlflowClient()
if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_name, version=model_details.version, key="has_signature", value=0)
else:
  client.set_model_version_tag(name=model_name, version=model_details.version, key="has_signature", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Demographic accuracy
# MAGIC 
# MAGIC How does the model perform across various slices of the customer base?d

# COMMAND ----------

import numpy as np
client = MlflowClient()
    
# Did not use SPARK dataframe, see comments in previous para
# features = features.withColumn('predictions', loaded_model(*features.columns)).toPandas()
 
df['accurate'] = np.where(df.churn == cust_predict.predictions, 1, 0)
 
# Check run tags for demographic columns and accuracy in each segment
try:
  demographics = run_info.data.tags['demographic_vars'].split(",")
  slices = df.groupby(demographics).accurate.agg(acc = 'sum', obs = lambda x:len(x), pct_acc = lambda x:sum(x)/len(x))
  
  # Threshold for passing on demographics is 55%
  demo_test = "pass" if slices['pct_acc'].any() > 0.55 else "fail"
  
  # Set tags in registry
  client.set_model_version_tag(name=model_name, version=model_details.version, key="demo_test", value=demo_test)
 
  print(slices)
except KeyError:
  print("KeyError: No demographics_vars tagged with this model version.")
  client.set_model_version_tag(name=model_name, version=model_details.version, key="demo_test", value="none")
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artifact check
# MAGIC Have artifacts being logged with the original model?

# COMMAND ----------

import os

client = MlflowClient()

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = mlflow.artifacts.download_artifacts(run_id = run_id, artifact_path = "")

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(name=model_name, version=model_details.version, key="has_artifacts", value=0)
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(name=model_name, version=model_details.version, key = "has_artifacts", value = 1)
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC 
# MAGIC Here's a summary of the testing results:

# COMMAND ----------

client = MlflowClient()
results = client.get_model_version(model_name, model_details.version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC ## Move to Production or None
# MAGIC 
# MAGIC If it fared well in testing, move to production. Otherwise, archive with a warning

# COMMAND ----------

client = MlflowClient()
# If any checks failed, reject and move to Archived
if '0' in results or 'fail' in results:
    client.transition_model_version_stage(model_details.name, model_details.version, "None")
    client.set_model_version_tag(name=model_name, version=model_version, key = "overall status", value = "Please check tags to reassess")
else:
    client.transition_model_version_stage(model_details.name, model_details.version, "Production")
    client.set_model_version_tag(name=model_name, version=model_details.version, key = "overall status", value = "Clear for prod use")
    print ("All tests passed! Clear for prod use.")
