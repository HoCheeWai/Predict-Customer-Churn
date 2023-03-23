# Databricks notebook source
# MAGIC %md
# MAGIC <img src="https://github.com/RafiKurlansik/laughing-garbanzo/blob/main/step2.png?raw=true">

# COMMAND ----------

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

import mlflow
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import lightgbm as lgb

mlflow.autolog()
#mlflow.sklearn.autolog(disable=True)
mlflow.sklearn.autolog(log_input_examples=True, silent=True)
mlflow.end_run()

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

