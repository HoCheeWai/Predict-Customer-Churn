# Databricks notebook source
# MAGIC %md
# MAGIC ## Churn Prediction Batch Inference
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step6.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load Model
# MAGIC 
# MAGIC Originally, loading as a Spark UDF to set us up for future scale. However, there was an error so did not use Spark Dataframe

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient
from databricks.feature_store import FeatureStoreClient

client = MlflowClient()
model_name = "telco_churn"
model_details = client.get_latest_versions(name = model_name, stages = ["Production"])
latest_version = model_details[0].version

try:
    data_source = client.get_run(model_details[0].run_id)
    data_source = run_info.data.tags['db_table']
except Exception:
    data_source = 'ibm_telco_churn.churn_features' # make an assumption
    pass   

print('Version {} of model {}'.format(latest_version, model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference

# COMMAND ----------

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read from feature store
fs = FeatureStoreClient()

features = fs.read_table(data_source)
features = features.withColumn("TotalCharges", features.TotalCharges.cast('double'))

loaded_model = mlflow.pyfunc.load_model(f'models:/{model_name}/{latest_version}')

df = features.toPandas()
X = df.drop(['churn', 'customerID'], axis=1)

try:
    lgb_pred = loaded_model.predict(X)
except Exception:
    print("Unable to predict on features.")
    pass 

cust_predict = df
cust_predict['predictions'] = lgb_pred

accuracy=accuracy_score(df['churn'], cust_predict['predictions'])
print('LightGBM Model accuracy score: {0:0.4f}'.format(accuracy))

display (cust_predict)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write to Delta Lake

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
cd_df = spark.createDataFrame(cust_predict)
display(cd_df)

cd_df.write.format("delta").mode("append").saveAsTable("ibm_telco_churn.churn_preds")
