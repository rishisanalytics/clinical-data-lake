# Databricks notebook source
# MAGIC %md
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC 
# MAGIC # Real World Evidence Data Analysis
# MAGIC ## 3. Distributed ML with MLFlow and Hyperopt
# MAGIC In this notebook, we train a model to predict whether a patient is at risk of a given condition, using the patient's encounter history and demographic information. 
# MAGIC 
# MAGIC <ol>
# MAGIC   <li> **Data**: We use the dataset in `rwd_hls` database that we created using simulated patient records.</li>
# MAGIC   <li> **Parameteres**: Users can specify the target condition (to be predicted), the number of comorbid conditions to include, number of days of record, and training/test split fraction.
# MAGIC   <li> **Model Training**: We use [*spark ml*](https://spark.apache.org/docs/1.2.2/ml-guide.html)'s' random forest algorithm for binary classification. Moreover we use [*hyperopt*](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/index.html#hyperopt) for distributed hyperparameter tuning </li>
# MAGIC   <li> **Model tracking and management**: Using [*MLFlow*](https://docs.databricks.com/applications/mlflow/index.html#mlflow), we track our training experiments and log the models for each run </li>
# MAGIC </ol>

# COMMAND ----------

# DBTITLE 1,add text box for input parameters
dbutils.widgets.text('condition', '', 'Condition to model')
dbutils.widgets.text('num_conditions', '10', '# of comorbidities to include')
dbutils.widgets.text('days', '90', '# of days to use')
dbutils.widgets.text('training_set_size', '70', '% of samples for training set')

# COMMAND ----------

import hyperopt as hp
from hyperopt import fmin, rand, tpe, hp, Trials, exceptions, space_eval, STATUS_OK

import mlflow
import mlflow.spark

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import Window
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Data Preparation
# MAGIC To create the training data, we need to extract a dataset with both positive (affected ) and negative (not affected) labels.

# COMMAND ----------

# DBTITLE 1,load data for training
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

# DBTITLE 1,create list of positive and negative samples
## create a list of patients that do not have the condition (negative control)

unaffected_patients = (
 
  patients
  .join(condition_patients,on=patients['Id'] == condition_patients['PATIENT'],how='left_anti')
  .limit(condition_patients.count())
  .select(F.col('Id').alias('PATIENT'))
)

# create a list of patients to include in training 
patients_to_study = condition_patients.union(unaffected_patients).cache()

# split dataste into train/test 
training_set_fraction = float(dbutils.widgets.get('training_set_size')) / 100.0

(train_patients, test_patients) = patients_to_study.randomSplit([training_set_fraction, 1.0 - training_set_fraction])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Feature Engineering
# MAGIC Now that we have the data that we need, we create a function that takes the list of patient's to inlcude, as well as an optional array of fitted indexers (which will be used for creating features in the test set) and outputs the dataset that will be used for calssification.

# COMMAND ----------

# DBTITLE 1,define a function for feature eng
def featurize_encounters(patients, string_indicers=None):

  # get the first encounter date within the dataset
  lowest_date = (
    encounters
    .select(encounters['START_TIME'])
    .orderBy(encounters['START_TIME'])
    .limit(1)
    .withColumnRenamed('START_TIME', 'EARLIEST_TIME')
  )
  
  # get the last encounter date within the dataset
  encounter_features = (
     encounters.join(patients, on='PATIENT')
    .where(encounters['REASONDESCRIPTION'].isNotNull())
    .crossJoin(lowest_date)
    .withColumn("day", F.datediff(F.col('START_TIME'), F.col('EARLIEST_TIME')))
    .withColumn("patient_age", F.datediff(F.col('START_TIME'), F.col('BIRTHDATE')))
  )
  
  # collect the list of comorbid conditions
  comorbidities = comorbid_conditions.collect()
  
  # now for each comorbid condition we add a feature column which indicates presense or absense of the condition for each patient in the training set
  idx = 0
  for comorbidity in comorbidities:
    encounter_features = encounter_features.withColumn("comorbidity_%d" % idx,encounter_features['REASONDESCRIPTION'].like('%' + comorbidity['REASONDESCRIPTION'] + '%')).cache()
    idx += 1
  
  string_index_cols = []
  strings_to_index = ['MARITAL', 'RACE', 'GENDER']
  
  # if the user specifies a list of string_indicers then those will be used (this is the case for when use it on the test data)
  if string_indicers:
    for model in string_indicers:
      encounter_features = model.transform(encounter_features)
    
    for string_to_index in strings_to_index:
      outCol = string_to_index + 'idx'
      string_index_cols.append(outCol)
      
  # creating an array of string indicers to transform categorical columns and adding transformed columns
  else:
    string_indicers = []
  
    for string_to_index in strings_to_index:
      outCol = string_to_index + 'idx'
      string_index_cols.append(outCol)
    
      si = StringIndexer(inputCol=string_to_index, outputCol=(outCol), handleInvalid='skip')
      model = si.fit(encounter_features)
      string_indicers.append(model)
      encounter_features = model.transform(encounter_features)
  
  # define a window function to include only records that are within the specified number of days 
  w = (
    Window.orderBy(encounter_features['day'])
    .partitionBy(encounter_features['PATIENT'])
    .rangeBetween(-int(dbutils.widgets.get('days')), -1)
  )
  # for each comorbidity add a column of the number of recent encounters
  comorbidity_cols = []
  for comorbidity_idx in range(idx):
    col_name = "recent_%d" % comorbidity_idx
    comorbidity_cols.append(col_name)
    
    encounter_features = encounter_features.withColumn(col_name, F.sum(F.col("comorbidity_%d" % comorbidity_idx).cast('int')).over(w)).\
      withColumn(col_name, F.expr("ifnull(%s, 0)" % col_name))
  
  # adding a column that indicates the number of recent encounters (within the specified number of days to use)
  encounter_features = encounter_features.withColumn("recent_encounters", F.count(F.lit(1)).over(w))
  
  # creating a vector of all features
  v = VectorAssembler(inputCols=comorbidity_cols + string_index_cols + ['ZIP'], outputCol='features', handleInvalid='skip')
  encounter_features = v.transform(encounter_features)
  
  encounter_features = encounter_features.withColumn('label', encounter_features['comorbidity_0'].cast('int'))
  
  return (encounter_features, string_indicers)

# COMMAND ----------

(training_encounters, string_indicers) = featurize_encounters(train_patients)
display(training_encounters)

# COMMAND ----------

display(training_encounters.select("features", "label"))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 3. Model Training, Tracking and Hyperparameter tunning
# MAGIC 
# MAGIC Now train a binary classifier (using random forests) for predicting the target condition. We use MLFlow for tracking and registering the model, and use hyperopt for distributed hyper parameter tuning.

# COMMAND ----------

# DBTITLE 1,training function
def train(params):
  with mlflow.start_run():
    
    impurity = params['impurity']
    max_depth = int(params['max_depth'])
    max_bins = int(params['max_bins'])
    
    mlflow.log_param("impurity", impurity)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_bins", max_bins)
    
    parameters = ['condition', 'num_conditions', 'days']
    for parameter in parameters:
      mlflow.log_param(parameter, dbutils.widgets.get(parameter))
  
    dt = DecisionTreeClassifier(impurity=impurity, maxDepth=max_depth, maxBins=max_bins)
  
    model = dt.fit(training_encounters)
    mlflow.spark.log_model(model, 'patient-trajectory')
  
    (testing_encounters, _) = featurize_encounters(test_patients, string_indicers=string_indicers)
  
    bce = BinaryClassificationEvaluator()
    aroc = bce.evaluate(model.transform(testing_encounters))
    mlflow.log_metric("area_under_ROC", aroc)
  
  return {'loss': -aroc, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### HyperOpt (Baysian optimization)
# MAGIC 
# MAGIC <div style="float:right"><img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/bayesian-model.png" style="height: 330px"/></div>
# MAGIC 
# MAGIC HyperOpt search accross your parameter space for the minimum loss of your model, using Bayesian optimization instead of a random walk

# COMMAND ----------

# DBTITLE 1,Define the hyperparameter grid
criteria = ['gini', 'entropy']
search_space = {
  'max_depth': hp.uniform('max_depth', 2, 25),
  'max_bins': hp.choice('max_bins', [8, 16, 32, 64]),
  'impurity': hp.choice('impurity', criteria)
}

# COMMAND ----------

# DBTITLE 1,Train the model and log the best model
#spark_trials = SparkTrials(parallelism=4)
spark_trials = Trials()

# The algoritm to perform the parameter search
algo = tpe.suggest

argmin = fmin(
  fn=train,
  space=search_space,
  algo=algo,
  max_evals=20,
  trials=spark_trials)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## MLFlow dashboard
# MAGIC 
# MAGIC Now if you click on `Runs` in the top right corner of the notebook, you can see a list of runs of the notebook wich keeps track of parameters used in training, as well as performance metric (area under ROC in this case). For more information on using MLFlow dashboard and runs on databricks see [this blog](https://databricks.com/blog/2019/04/30/introducing-mlflow-run-sidebar-in-databricks-notebooks.html).