# Databricks notebook source
# MAGIC %md
# MAGIC ### Managing the model lifecycle with Model Registry
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step4.png?raw=true">
# MAGIC 
# MAGIC One of the primary challenges among data scientists and ML engineers is the absence of a central repository for models, their versions, and the means to manage them throughout their lifecycle.  
# MAGIC 
# MAGIC [The MLflow Model Registry](https://docs.databricks.com/applications/mlflow/model-registry.html) addresses this challenge and enables members of the data team to:
# MAGIC <br><br>
# MAGIC * **Discover** registered models, current stage in model development, experiment runs, and associated code with a registered model
# MAGIC * **Transition** models to different stages of their lifecycle
# MAGIC * **Deploy** different versions of a registered model in different stages, offering MLOps engineers ability to deploy and conduct testing of different model versions
# MAGIC * **Test** models in an automated fashion
# MAGIC * **Document** models throughout their lifecycle
# MAGIC * **Secure** access and permission for model registrations, transitions or modifications
# MAGIC 
# MAGIC <!--<img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg"> -->

# COMMAND ----------

import mlflow.sklearn
from mlflow.entities import ViewType

# For testing, allow manual run to collect parameters
dbutils.widgets.text('model_registry_secret_scope', '')
dbutils.widgets.text('model_registry_secret_key_prefix', '')
dbutils.widgets.text('run_id', '')
 
mr_scope = str(dbutils.widgets.get('model_registry_secret_scope'))
mr_key = str(dbutils.widgets.get('model_registry_secret_key_prefix'))
run_id = str(dbutils.widgets.get('run_id'))

if len(mr_scope.strip()) == 0:
    mr_scope = 'prod' # default to prod registry
if len(mr_key.strip()) == 0:
    mr_key = 'prod'

# Create the URIs to use to work with the remote Feature Store and Model Registry
model_registry_uri = f'databricks://{mr_scope}:{mr_key}' if mr_scope and mr_key else None
       
if len(run_id.strip()) == 0:
   run_id = dbutils.jobs.taskValues.get(taskKey = "Train_model", key = "run_id", debugValue = 0)

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


# COMMAND ----------

# MAGIC %md
# MAGIC ### How to Use the Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters. During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.  Think of this as **committing** the model to the registry, much as you would commit code to a version control system.  
# MAGIC 
# MAGIC The registry defines several model stages: `None`, `Staging`, `Production`, and `Archived`. Each stage has a unique meaning. For example, `Staging` is meant for model testing, while `Production` is for models that have completed the testing or review processes and have been deployed to applications. 
# MAGIC 
# MAGIC Users with appropriate permissions can transition models between stages.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Promote to Registry
# MAGIC ```
# MAGIC model_name = "Example"
# MAGIC 
# MAGIC model_uri = f"runs:/{ mlflow_run.info.run_id }/model"
# MAGIC registered_model_version = mlflow.register_model(model_uri, model_name)
# MAGIC ```

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "telco_churn"
model_uri = f"runs:/{run_id}/model"
mlflow.set_registry_uri(model_registry_uri)
model_details = mlflow.register_model(model_uri, model_name)

client.set_tag(run_id, key='db_table', value='ibm_telco_churn.churn_features')
client.set_tag(run_id, key='demographic_vars', value='SeniorCitizen,gender_Female')

# COMMAND ----------

# MAGIC %md
# MAGIC At this point the model will be in `None` stage.  Let's update the description before moving it to `Staging`.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Update Description

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

client = MlflowClient()
model_version_details = client.get_model_version(name=model_name, version=model_details.version)

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

# MAGIC %md
# MAGIC #### Transition to Staging
# MAGIC 
# MAGIC <!--<img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/webhooks2.png?raw=true" width = 800> -->

# COMMAND ----------

# Using mlflow REST API, a request for transition to staging was made. However, this seem to 
# have challenges updating the model registry in different work.
# Using python API, the transition is direct, there is no request
# There is also no equivalent as the REST API to add comments to a version
client.transition_model_version_stage(model_details.name, model_details.version, "staging")
