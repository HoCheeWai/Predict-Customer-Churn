# Databricks notebook source
# MAGIC %sh
# MAGIC wget https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
# MAGIC pwd

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS ibm_telco_churn;
# MAGIC SHOW DATABASES;

# COMMAND ----------

import pandas as pd
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql import DataFrameWriter

pdf = pd.read_csv("/Workspace/Repos/ho_chee_wai@rp.edu.sg/Predict-Customer-Churn/Telco-Customer-Churn.csv")

spark = SparkSession.builder.getOrCreate()
df = spark.createDataFrame(pdf)
display(df)

df.write.mode("overwrite").saveAsTable(name='ibm_telco_churn.bronze_customers', format='delta')
