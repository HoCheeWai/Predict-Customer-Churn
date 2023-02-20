# Databricks notebook source
# MAGIC %md
# MAGIC ## Churn Prediction Feature Engineering
# MAGIC 
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step1.png?raw=true">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC This is a fairly clean dataset so we'll just do some one-hot encoding, and clean up the column names afterward.

# COMMAND ----------

# DBTITLE 1,Read in Bronze Delta table using Spark
# Read into Spark
telcoDF = spark.table("ibm_telco_churn.bronze_customers")

display(telcoDF)

# COMMAND ----------

# MAGIC %md
# MAGIC Using `koalas` allows us to scale `pandas` code.

# COMMAND ----------

# DBTITLE 1,Define featurization function
import pyspark.pandas as ps

def compute_churn_features(data):
  
  # convert from sql to pandas dataframe
  data = data.pandas_api()
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'Partner', 'Dependents',
                                 'PhoneService', 'MultipleLines', 'InternetService',
                                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                 'Contract', 'PaperlessBilling', 'PaymentMethod'],dtype = 'int64')
  
  # Convert label to int and rename column
  data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
  data = data.astype({'Churn': 'int32'})
  data = data.rename(columns = {'Churn': 'churn'})
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
    
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# DBTITLE 1,Write features to the feature store
from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

churn_features_df = compute_churn_features(telcoDF)

churn_feature_table = fs.create_table(
  name='ibm_telco_churn.churn_features',
  primary_keys='customerID',
  schema=churn_features_df.spark.schema(),
  description='These features are derived from the ibm_telco_churn.bronze_customers table in the lakehouse.  I created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
)

fs.write_table(df=churn_features_df.to_spark(), name='ibm_telco_churn.churn_features', mode='overwrite')

# COMMAND ----------

