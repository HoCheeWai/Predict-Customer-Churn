# Databricks notebook source
# MAGIC %md
# MAGIC ## Model Tests
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step5.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fetch Model in Transition

# COMMAND ----------

import mlflow, json
from mlflow.tracking import MlflowClient
from databricks.feature_store import FeatureStoreClient

# For testing, allow manual run to collect parameters
dbutils.widgets.text('model_registry_secret_scope', '')
dbutils.widgets.text('model_registry_secret_key_prefix', '')
dbutils.widgets.text('expt_run_id', '')
 
mr_scope = str(dbutils.widgets.get('model_registry_secret_scope'))
mr_key = str(dbutils.widgets.get('model_registry_secret_key_prefix'))
run_id = str(dbutils.widgets.get('expt_run_id'))

if len(mr_scope.strip()) == 0:
    mr_scope = 'prod' # default to prod registry
if len(mr_key.strip()) == 0:
    mr_key = 'prod'

# Create the URIs to use to work with the remote Model Registry
model_registry_uri = f'databricks://{mr_scope}:{mr_key}' if mr_scope and mr_key else None

client = MlflowClient(registry_uri = model_registry_uri)
fs = FeatureStoreClient()
       
if len(run_id.strip()) == 0:
   run_id = dbutils.jobs.taskValues.get(taskKey = "Train", key = "run_id", debugValue = 0)

if run_id == 0 or run_id is None:
    all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
    runs = mlflow.search_runs(
        experiment_ids = all_experiments,
        filter_string = 'metrics.test_accuracy_score > 0.78 and status = "FINISHED" and ' +
            'tags.mlflow.runName = "lgbm predict churn"',
        run_view_type = ViewType.ACTIVE_ONLY,
    )
    run_id = runs.loc[runs['end_time'].idxmax()]['run_id']

display (run_id)

# Get the latest version
model_name = "telco_churn"
mlflow.set_registry_uri(model_registry_uri)
model_details = client.get_latest_versions(name = model_name, stages = ["staging"])
run_info = client.get_run(run_id)

model_version = 0
model_latest_run_id = 0
for m in model_details:
    print("\nname: {}".format(m.name))
    print("latest version: {}".format(m.version))
    print("run_id: {}".format(m.run_id))
    print("current_stage: {}\n".format(m.current_stage))
    model_version =  m.version
    model_latest_run_id = m.run_id
    model_desc =  m.description

display (run_info)

if run_id != model_latest_run_id:
    print("\nLatest run_id from model_registry {} and run_id from parameters {} differs".format(latest_run_id, run_id))
    print("\n will proceed with the run_id from parameters")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Validate prediction

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read from feature store
data_source = run_info.data.tags['db_table']
features = fs.read_table(data_source)
features = features.withColumn("TotalCharges", features.TotalCharges.cast('double'))
display(features)

loaded_model = mlflow.pyfunc.load_model(f'models:/{model_name}/Staging')

df = features.toPandas()
X = df.drop(['churn', 'customerID'], axis=1)
y = df.churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

try:
    lgb_pred = loaded_model.predict(X_test)
    accuracy=accuracy_score(lgb_pred, y_test)
    print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, lgb_pred)))
    client.set_model_version_tag(name=model_name, version=model_version, key="predicts", value=1)
except Exception:
    print("Unable to predict on features.")
    client.set_model_version_tag(name=model_name, version=model_version, key="predicts", value=0)
    pass 

# Load model as a Spark UDF
# Not able to predict using Spark. It says, IllegalStateException: Cmd gets aborted by kernel which may be in a bad state
# There is some issue with the seniorCitizen column. Screen captures at "Databricks_auto_retrain.docx"
# It could be because the mode was NOT trained using the feature store
# model_uri = f'models:/{model_name}/{model_version}' 
# loaded_model = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
# Predict on a Spark DataFrame

# try:
#    display(features.withColumn('predictions', loaded_model(*features.columns)))
#    client.set_model_version_tag(name=model_name, version=model_version, key="predicts", value=1)
# except Exception: 
#    print("Unable to predict on features.")
#    client.set_model_version_tag(name=model_name, version=model_version, key="predicts", value=0)
#   pass

# COMMAND ----------

# MAGIC %md
# MAGIC #### Signature check
# MAGIC 
# MAGIC When working with ML models you often need to know some basic functional properties of the model at hand, such as “What inputs does it expect?” and “What output does it produce?”.  The model **signature** defines the schema of a model’s inputs and outputs. Model inputs and outputs can be either column-based or tensor-based. 
# MAGIC 
# MAGIC See [here](https://mlflow.org/docs/latest/models.html#signature-enforcement) for more details.

# COMMAND ----------

if not loaded_model.metadata.signature:
  print("This model version is missing a signature.  Please push a new version with a signature!  See https://mlflow.org/docs/latest/models.html#model-metadata for more details.")
  client.set_model_version_tag(name=model_name, version=model_version, key="has_signature", value=0)
else:
  client.set_model_version_tag(name=model_name, version=model_version, key="has_signature", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Demographic accuracy
# MAGIC 
# MAGIC How does the model perform across various slices of the customer base?

# COMMAND ----------

import numpy as np
df['prediction'] = loaded_model.predict(df)

# Did not use SPARK dataframe, see comments in previous para
# features = features.withColumn('predictions', loaded_model(*features.columns)).toPandas()

df['accurate'] = np.where(df.churn == df.prediction, 1, 0)

# Check run tags for demographic columns and accuracy in each segment
try:
  demographics = run_info.data.tags['demographic_vars'].split(",")
  slices = df.groupby(demographics).accurate.agg(acc = 'sum', obs = lambda x:len(x), pct_acc = lambda x:sum(x)/len(x))
  
  # Threshold for passing on demographics is 55%
  demo_test = "pass" if slices['pct_acc'].any() > 0.55 else "fail"
  
  # Set tags in registry
  client.set_model_version_tag(name=model_name, version=model_version, key="demo_test", value=demo_test)

  print(slices)
except KeyError:
  print("KeyError: No demographics_vars tagged with this model version.")
  client.set_model_version_tag(name=model_name, version=model_version, key="demo_test", value="none")
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## Documentation 
# MAGIC Is the model documented visually and in plain english?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Description check
# MAGIC 
# MAGIC Has the data scientist provided a description of the model being submitted?

# COMMAND ----------

if not model_desc:
  client.set_model_version_tag(name=model_name, version=model_version, key="has_description", value=0)
  print("Did you forget to add a description?")
elif not len(model_desc) > 20:
  client.set_model_version_tag(name=model_name, version=model_version, key="has_description", value=0)
  print("Your description is too basic, sorry.  Please resubmit with more detail (40 char min).")
else:
  client.set_model_version_tag(name=model_name, version=model_version, key="has_description", value=1)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artifact check
# MAGIC Has the data scientist logged supplemental artifacts along with the original model?

# COMMAND ----------

import os

# Create local directory 
local_dir = "/tmp/model_artifacts"
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

# Download artifacts from tracking server - no need to specify DBFS path here
local_path = mlflow.artifacts.download_artifacts(run_id = run_id, artifact_path = "")

# Tag model version as possessing artifacts or not
if not os.listdir(local_path):
  client.set_model_version_tag(name=model_name, version=model_version, key="has_artifacts", value=0)
  print("There are no artifacts associated with this model.  Please include some data visualization or data profiling.  MLflow supports HTML, .png, and more.")
else:
  client.set_model_version_tag(name=model_name, version=model_version, key = "has_artifacts", value = 1)
  print("Artifacts downloaded in: {}".format(local_path))
  print("Artifacts: {}".format(os.listdir(local_path)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC 
# MAGIC Here's a summary of the testing results:

# COMMAND ----------

results = client.get_model_version(model_name, model_version)
results.tags

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leave in Staging or move to Archived
# MAGIC 
# MAGIC The next phase of this models' lifecycle will be to `Staging` or `Archived`, depending on how it fared in testing.

# COMMAND ----------

# If any checks failed, reject and move to Archived
if '0' in results or 'fail' in results:
    client.transition_model_version_stage(model_details.name, model_details.version, "archived")
    client.set_model_version_tag(name=model_name, version=model_version, key = "overall status", value = "Please check tags to reassess")
else:
    client.set_model_version_tag(name=model_name, version=model_version, key = "overall status", value = "May proceed to cut a release")
    print ("All tests passed! May proceed to cut a release.")
