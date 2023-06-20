# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - Commodity load job

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import sys
import os
import sqls
import datetime

from integration_tests.quality_utils import SbnNutterFixture

from utils import *
from utils.constants import Zone, Table

from modules.utils.constants import MLTable

class CommodityLoadFixture(SbnNutterFixture):
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_dim_directory_commodity_load_job"
        self.import_module = "dim_directory_commodity_load_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.RAW.value, f'{os.path.dirname(sqls.__file__)}/raw')
        database_utils.execute_ddl_sql(Zone.CONSUMPTION.value, f'{os.path.dirname(sqls.__file__)}/consumption')
        database_utils.create_all_raw_tables()

    def before_dim_directory_commodity_load_1_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        # Add the test data to the source table
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_earliest = timestamp_earliest - datetime.timedelta(
            microseconds=1
        )
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)
        commodity_test_df = spark.createDataFrame([
            (100, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_current),
            (101, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_before_earliest),
            (102, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_after_current),
            (103, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_current)
        ],schema='ID long, PARENT long, CODE long, NAME string, LEVEL_NAME string, VALID int, SUBCODE long, SQID long, IS_STANDARD int, _ID long, _CHANGED_DATETIME timestamp, _DELTA_CREATED_ON timestamp, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.RAW.value, MLTable.RAW_COMMODITY.value).alias("t")
            .merge(commodity_test_df.alias("s"), "t.ID = s.ID")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def run_dim_directory_commodity_load_1_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.import_module, "tasklist": self.import_module},
        )

    def assertion_dim_directory_commodity_load_1_fisrt_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(
            Zone.CONSUMPTION.value, MLTable.CONSUMPTION_COMMODITY.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID"
        ).collect()
        assert len(target_table_list) == 4
        assert len(target_table_list[0]) == 13
        assert target_table_list[0]['CODE'] == '20121318'
        assert target_table_list[0]['LEVEL_NAME'] == 'Commodity'

        # Assert job config table
        self.assert_job_config_table(self.import_module)

    def before_dim_directory_commodity_load_2_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_earliest = timestamp_earliest - datetime.timedelta(
            microseconds=1
        )
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)
        commodity_test_df = spark.createDataFrame([
            (100, 7093, 20121318, 'Sand detectors', 'Commodity1', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_current),
            (101, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_before_earliest),
            (102, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_after_current),
            (104, 7093, 20121318, 'Sand detectors', 'Commodity', 1, 18, 12795, 1, 100, timestamp_current, timestamp_current, timestamp_current)
        ],schema='ID long, PARENT long, CODE long, NAME string, LEVEL_NAME string, VALID int, SUBCODE long, SQID long, IS_STANDARD int, _ID long, _CHANGED_DATETIME timestamp, _DELTA_CREATED_ON timestamp, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.RAW.value, MLTable.RAW_COMMODITY.value).alias("t")
            .merge(commodity_test_df.alias("s"), "t.ID = s.ID")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

    def run_dim_directory_commodity_load_2_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.import_module, "tasklist": self.import_module},
        )

    def assertion_dim_directory_commodity_load_2_fisrt_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(
            Zone.CONSUMPTION.value, MLTable.CONSUMPTION_COMMODITY.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID"
        ).collect()
        assert len(target_table_list) == 5
        assert len(target_table_list[0]) == 13
        assert target_table_list[0]['CODE'] == '20121318'
        assert target_table_list[0]['LEVEL_NAME'] == 'Commodity1'

        # Assert job config table
        self.assert_job_config_table(self.import_module)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = CommodityLoadFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
