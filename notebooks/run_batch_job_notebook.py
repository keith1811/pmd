# Databricks notebook source
# MAGIC %md
# MAGIC ### Generic batch job notebook

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parameters

# COMMAND ----------

import datetime

dbutils.widgets.removeAll()
dbutils.widgets.text("import_module", "")
dbutils.widgets.text("skip_set_offset_timestamp", "false")
dbutils.widgets.text("integration_test_mode", "false")
dbutils.widgets.text("commit_id", "commit_id")
dbutils.widgets.text(
    "commit_timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
)
dbutils.widgets.text("starting_timestamp_passed_in", "")
dbutils.widgets.text("ending_timestamp_passed_in", "")
dbutils.widgets.text("new_columns_for_test", "")
dbutils.widgets.text("tasklist", "")
dbutils.widgets.text("need_archive", "false")
dbutils.widgets.text("batch_size", "")
dbutils.widgets.text("reprocess_module_list", "")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import datetime
import sys
import importlib

sys.path.append("../modules")
from utils import sbnutils
from utils import batch_utils
from modules.utils.workflow_utils import if_run_task

# Initialize spark session
sbnutils.init_spark()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get offset timestamp before running the job notebook

# COMMAND ----------

starting_timestamp = dbutils.widgets.get("starting_timestamp_passed_in")
ending_timestamp = dbutils.widgets.get("ending_timestamp_passed_in")

if not starting_timestamp:
    starting_timestamp = batch_utils.get_batch_offset_timestamp()

if not ending_timestamp:
    ending_timestamp = datetime.datetime.now()

dbutils.widgets.text("starting_timestamp", str(starting_timestamp))
dbutils.widgets.text("ending_timestamp", str(ending_timestamp))

import_module = dbutils.widgets.get("import_module")
sbnutils.log_info(f"Batch job '{import_module}' started.")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the batch job

# COMMAND ----------
import_module = dbutils.widgets.get("import_module")
tasklist = dbutils.widgets.get("tasklist")
if_run = if_run_task(tasklist, import_module)
try:
    if if_run:
        imported_module = importlib.import_module(import_module)
        imported_module.main()
        sbnutils.log_info(f"Batch job '{import_module}' finished.")
except Exception as e:
    sbnutils.log_error(f"Run the batch job '{import_module}' - exception: {e}")
    dbutils.widgets.removeAll()
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set offset timestamp after running the job notebook

# COMMAND ----------

if dbutils.widgets.get("skip_set_offset_timestamp").lower() != "true" and if_run:
    batch_utils.set_batch_offset_timestamp(ending_timestamp)

sbnutils.log_info(f"Batch job '{import_module}' succeeded.")
dbutils.widgets.removeAll()
