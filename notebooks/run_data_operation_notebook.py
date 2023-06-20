# Databricks notebook source
# MAGIC %md
# MAGIC ### Data Operation notebook

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
!cp ../requirements-ml.txt ~/.
%pip install -r ~/requirements-ml.txt
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import utils.sbnutils as sbnutils
import datetime
from utils.constants import Zone
from utils.constants import Table

sbnutils.init_spark()

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------

dbutils.widgets.text("integration_test_mode", "false")
dbutils.widgets.text("operationType", "")
dbutils.widgets.text("zone", "")
dbutils.widgets.text("table", "")
dbutils.widgets.text("insertColumns", "")
dbutils.widgets.text("insertValues", "")
dbutils.widgets.text("updateExpr", "")
dbutils.widgets.text("condition", "")

# COMMAND ----------

operationType = dbutils.widgets.get("operationType")
zone = dbutils.widgets.get("zone")
table = dbutils.widgets.get("table")
insertColumns = dbutils.widgets.get("insertColumns")
insertValues = dbutils.widgets.get("insertValues")
updateExpr = dbutils.widgets.get("updateExpr")
condition = dbutils.widgets.get("condition")

table_location = sbnutils.get_table_full_name(zone, table)
table_storage_account_location = sbnutils.get_table_storage_location(zone=zone, table_name=table)
print(table_location)
print(table_storage_account_location)

# COMMAND ----------

if operationType.lower() == "update":
    sql = f"""
        UPDATE {table_location} SET {updateExpr} WHERE {condition}
        """
elif operationType.lower() == "insert":
    sql = f"""
        INSERT INTO {table_location} {insertColumns} VALUES {insertValues}
        """
elif operationType.lower() == "delete":
    sql = f"""
        DELETE FROM {table_location} WHERE {condition}
        """
elif operationType.lower() == "drop":
    sql = f"""
        DROP TABLE IF EXISTS {table_location}
        """
    dbutils.fs.rm(table_storage_account_location, True)

print(sql)
spark.sql(sql)

# COMMAND ----------

dbutils.widgets.removeAll()
