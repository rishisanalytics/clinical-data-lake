# Databricks notebook source
# DBTITLE 1,Dependencies
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

# DBTITLE 1,Featurise
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

# DBTITLE 1,Train function
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

def testNotebook(x):
  print("Test passed : " + x)