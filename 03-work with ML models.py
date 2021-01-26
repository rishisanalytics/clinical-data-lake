# Databricks notebook source
# MAGIC %md
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC 
# MAGIC # Real World Evidence Data Analysis
# MAGIC ## 4. Model Exploration using MLFlow
# MAGIC In this notebook, we explore models in different stages of the ML lifecycle by loading and applying them to patients.
# MAGIC 
# MAGIC <ol>
# MAGIC   <li> **Data**: We use the dataset in `rwd_hls` database that we created using simulated patient records.</li>
# MAGIC   <li> **Parameteres**: Users can specify the target condition (to be predicted), the number of comorbid conditions to include, number of days of record, and training/test split fraction.
# MAGIC </ol>

# COMMAND ----------

# DBTITLE 1,load functions required to data
# MAGIC %run "./include/featurise"

# COMMAND ----------

# DBTITLE 1,Load required libraries
import mlflow
print(mlflow.__version__)
import mlflow.spark

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

# COMMAND ----------

run = MlflowClient().search_runs(
  experiment_ids="6879202",
  filter_string="",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.area_under_ROC DESC"]
)[0]

print("run id : " + run.info.run_id)
print("run metric AROC : {}".format(run.data.metrics['area_under_ROC']))

# COMMAND ----------

# DBTITLE 1,input parameters
dbutils.widgets.text('condition', '', 'Condition to model')
dbutils.widgets.text('num_conditions', '10', '# of comorbidities to include')
dbutils.widgets.text('days', '90', '# of days to use')

# COMMAND ----------

# MAGIC %md
# MAGIC ####Table History
# MAGIC Let's optimise our encounters table to speed up downstream queries

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC OPTIMIZE rwd_hls.encounters ZORDER BY PATIENT, REASONDESCRIPTION

# COMMAND ----------

# MAGIC %sql
# MAGIC --Explore the history of changes made to the table
# MAGIC DESCRIBE history rwd_hls.encounters

# COMMAND ----------

# DBTITLE 1,build a list of comorbid conditions, which is used for featurisation
delta_path="/tmp/rishi.ghose@databricks.com/rwe-ehr/delta"
# load data from rwd database
patients = spark.read.load(delta_path+'/patients')
encounters = spark.read.load(delta_path+'/patient_encounters')

# get the list of patients with the target condition (cases)
condition_patients = spark.sql("SELECT DISTINCT PATIENT FROM rwd_hls.patient_encounters WHERE lower(REASONDESCRIPTION) LIKE '%" + dbutils.widgets.get('condition') + "%'")

#create a dataframe of comorbid conditions
comorbid_conditions = (
 
  condition_patients.join(encounters, ['PATIENT'])
  .where(F.col('REASONDESCRIPTION').isNotNull())
  .dropDuplicates(['PATIENT', 'REASONDESCRIPTION'])
  .groupBy('REASONDESCRIPTION').count()
  .orderBy('count', ascending=False)
  .limit(int(dbutils.widgets.get('num_conditions')))
)

display(comorbid_conditions)

# COMMAND ----------

#get list of all patients, we will apply the model to this list
allPatients = patients.selectExpr("id as PATIENT")

# COMMAND ----------

#featurise the patient based on their encounter history
(encounterDF, string_indicers) = featurize_encounters(allPatients)
display(encounterDF.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ####Model Scoring

# COMMAND ----------

#load a specific version of the model from the model registry
loaded_model = mlflow.spark.load_model("models:/rishi_rwd_ehr/6")

#score the patient data with the model
scored_df = loaded_model.transform(encounterDF)

display(scored_df)

# COMMAND ----------

display(
  scored_df.where("prediction = 1").groupBy("MARITAL", "RACE", "ETHNICITY", "GENDER", "prediction").count()
)

# COMMAND ----------

# We retrieve some information about the model
client = MlflowClient()

model_name = "rishi_rwd_ehr"
prod_model = client.get_latest_versions(name = model_name, stages = ["Production"])[0]
print(prod_model.name)
print("Model version: ", prod_model.version)
print("Model source: ", prod_model.source)

# COMMAND ----------

# We retrieve the production model from the registry
loaded_model = mlflow.spark.load_model("models:/{}/{}".format(model_name, prod_model.version))

#We apply the production model to our dataframe 
results_df = loaded_model.transform(encounterDF)

# COMMAND ----------

from pyspark.sql.functions import lit
from pyspark.sql import functions as F

# We attach the model name, version and source to be able to keep track of what model was used. 
results_df = (results_df.withColumn("model_name", lit(prod_model.name))
                            .withColumn("model_version", lit(prod_model.version))
                            .withColumn("model_source", lit(prod_model.source))
                            .withColumn("model_run_timestamp", lit(F.current_timestamp()))
             )
                        

display(
  results_df.limit(5)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Other options for tracking an experiment

# COMMAND ----------

experimentID = '6879202'

# COMMAND ----------

# You can retrieve data from the mlflow ui using the following spark read syntax
runs = spark.read.format("mlflow-experiment").load(experimentID)
display(runs)

# COMMAND ----------

# You can tidy the dataframe, for instance using below code
runs = runs.withColumn('metrics_aroc', runs.metrics.area_under_ROC)

runs = runs.withColumn('impurity', runs.params.impurity)
runs = runs.withColumn('max_depth', runs.params.max_depth)
runs = runs.withColumn('max_bins', runs.params.max_bins)

runs = runs.drop('metrics').drop('params').drop('tags')

display(runs)

# COMMAND ----------

from pyspark.sql.functions import date_format, col

max_aroc = runs.agg({"metrics_aroc": "max"}).collect()[0][0]
print(max_aroc)

# COMMAND ----------

from pyspark.sql.functions import col

best_run = runs.filter(col('metrics_aroc')==max_aroc).select("run_id", "artifact_uri")
display(best_run)