# Databricks notebook source
# MAGIC %md
# MAGIC ##<img src="https://databricks.com/wp-content/themes/databricks/assets/images/header_logo_2x.png" alt="logo" width="150"/> 
# MAGIC 
# MAGIC # Real World Evidence Data Analysis
# MAGIC ## 2. Data visualization and dashboarding with R
# MAGIC In this notebook we create a simple dashboard that visualizes top conditions in the database (defined by a user-specified parameter) and also
# MAGIC performs a statistical test of significance of correalton between any two conditions specified by the user.

# COMMAND ----------

library(SparkR,stats)

encounters <- sql("SELECT * FROM rwd_hls.patient_encounters")
patients <- sql("SELECT * FROM rwd_hls.patients")

# COMMAND ----------

# MAGIC %md
# MAGIC Now we add [widgets]((https://docs.databricks.com/notebooks/widgets.html#widget-api) to the notebook 

# COMMAND ----------

dbutils.widgets.text('num_conditions', '20', '# of conditions to display')
dbutils.widgets.text('first_condition', 'chronic pain', 'First condition to test')
dbutils.widgets.text('second_condition', 'drug overdose', 'Second condition to test')

# COMMAND ----------

# DBTITLE 1,Total condition count
display(
  limit(
    arrange(sql("SELECT REASONDESCRIPTION as condition, count(*) AS count FROM rwd_hls.patient_encounters WHERE REASONDESCRIPTION IS NOT NULL GROUP BY REASONDESCRIPTION"),'count', decreasing = TRUE)
  , as.numeric(dbutils.widgets.get('num_conditions')))
)

# COMMAND ----------

# DBTITLE 1,Condition count by unique patient
display(
  limit(
    arrange(
      count(
        groupBy(
          sql(
            "SELECT DISTINCT PATIENT, REASONDESCRIPTION as condition FROM rwd_hls.patient_encounters WHERE REASONDESCRIPTION IS NOT NULL"
          ),
          'condition')), 'count', decreasing = TRUE
    )
    , as.numeric(dbutils.widgets.get('num_conditions'))
  )
)

# COMMAND ----------

# MAGIC %md # Find the count of user-specified conditions in the dataset

# COMMAND ----------

first_condition<-dbutils.widgets.get('first_condition')
second_condition<-dbutils.widgets.get('second_condition')
query <- paste("
  SELECT REASONDESCRIPTION, count('*') as count FROM rwd_hls.patient_encounters WHERE lower(REASONDESCRIPTION) LIKE '%",
  first_condition,"%'", "or lower(REASONDESCRIPTION) LIKE '%",second_condition, "%'","group by 1",sep = '')

display(sql(query))

# COMMAND ----------

# DBTITLE 1,Comorbid conditions
first_condition_patients <- (sql(paste("SELECT DISTINCT PATIENT FROM rwd_hls.patient_encounters WHERE lower(REASONDESCRIPTION) LIKE '%", dbutils.widgets.get('first_condition'), "%'", sep = '')))

display(
  limit(
    arrange(
      filter(
        count(
          groupBy(
            dropDuplicates(
              join(
                first_condition_patients, encounters, joinExpr=(first_condition_patients$PATIENT == encounters$PATIENT)), 'PATIENT', 'REASONDESCRIPTION')
            , 'REASONDESCRIPTION')
        ),
        paste("lower(REASONDESCRIPTION) NOT LIKE '%", dbutils.widgets.get('first_condition'), "%'", sep='')
      ),
      'count', decreasing = TRUE
    ),
    as.numeric(dbutils.widgets.get('num_conditions'))
  )
)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Investigate association between selected conditions
# MAGIC 
# MAGIC We can perform a chisqure test of association to see if there is any significance correlation between co-occurnace of conditions among patients in this dataset:

# COMMAND ----------

# List of patients with the first condition
first_condition_patients <- (sql(paste("SELECT DISTINCT PATIENT FROM rwd_hls.patient_encounters WHERE lower(REASONDESCRIPTION) LIKE '%", dbutils.widgets.get('first_condition'), "%'", sep = '')))
first_condition <- selectExpr(first_condition_patients, 'PATIENT', paste("'", dbutils.widgets.get('first_condition'), "' AS firstCondition", sep = ''))

# List of patients with the second condition
second_condition_patients <- (sql(paste("SELECT DISTINCT PATIENT FROM rwd_hls.patient_encounters WHERE lower(REASONDESCRIPTION) LIKE '%", dbutils.widgets.get('second_condition'), "%'", sep = '')))
second_condition <- selectExpr(second_condition_patients, 'PATIENT', paste("'", dbutils.widgets.get('second_condition'), "' AS secondCondition", sep = ''))

# create a table indicating co-occurnace of the two conditions
patients_conditions <- selectExpr(join(join(patients, first_condition, joinExpr = (patients$Id == first_condition$PATIENT), joinType = 'left_outer'), second_condition, joinExpr = (patients$Id == second_condition$PATIENT), joinType = 'left_outer'), "ifnull(firstCondition, 'none') AS first", "ifnull(secondCondition, 'none') AS second")

# collect the spark dataframe into a R base dataframe

patients_df <- collect(patients_conditions)

# perform chisqure test
chisq <- chisq.test(table(patients_df))

# print the resulst of the test
if (chisq$p.value < 5e-2) {
  displayHTML(sprintf(paste("<h2>the p-value is <i style=color:Red;>%e</i> (< 0.05), which supports a significant correlation between", dbutils.widgets.get('first_condition'), "and", dbutils.widgets.get('second_condition'), "</h2>"), chisq$p.value))
} else {
  displayHTML(sprintf(paste("<h2>the p-value is <i style=color:Red;>%e</i> (>0.05), which does not indicate enough support for correlation between", dbutils.widgets.get('first_condition'), "and", dbutils.widgets.get('second_condition'), "</h2>"), chisq$p.value))
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a dashboard
# MAGIC Now you can turn this notebook to a dashboard following [these instructions](https://docs.databricks.com/notebooks/dashboards.html#dashboards)