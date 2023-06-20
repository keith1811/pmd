# Databricks notebook source
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
from utils.constants import Zone, Table, DELTA_CREATED_FIELD, DELTA_UPDATED_FIELD
from pyspark.sql.functions import current_timestamp
from modules.utils.config_utils import get_model_name, get_model_version, get_model_stage

# COMMAND ----------

class PoItemRecordModelFixture(SbnNutterFixture):

    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """

        case_name = "it_ml_supplier_po_item_record_model_job"
        self.ml_supplier_po_item_record_model_module = "ml_supplier_po_item_record_model_module"
        self.test_model_uuid = ""
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        SbnNutterFixture.__init__(self, case_name)
        self.env = sbnutils._get_env()
        self.model_name = get_model_name(env=self.env)
        self.model_version = get_model_version(env=self.env)
        self.model_stage = get_model_stage(env = self.env)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')



    def before_add_table_for_model_info_1_first_round(self):
        # Simulate a table with model information in it 

        # Get target_table
        target_table = sbnutils.get_delta_table(Zone.GENERAL.value, "ml_model_info")

        # Create mock dataframe
        MODEL_INFO_SCHEMA = """
        UUID String, NAME String, VERSION int, STAGE String, _DELTA_CREATED_ON Timestamp, _DELTA_UPDATED_ON Timestamp"""
        timestamp_current = datetime.datetime.now()
        mock_df = spark.createDataFrame([
            ("f18206b8-cf72-11ed-ba7d-546cebfdedbd", "zzztest_po_clf", 27, "Staging", timestamp_current, timestamp_current),
            ("8d4f2426-cf7a-11ed-87fd-546cebfdedbd", "zzztest_po_reg", 28, "Dev", timestamp_current, timestamp_current)
        ], schema=MODEL_INFO_SCHEMA)
        # mock_df.show()
        
        # Write mock data into table
        (
            sbnutils.get_delta_table(Zone.GENERAL.value, 'ml_model_info').alias("t")
                .merge(mock_df.alias("s"), "t.NAME = s.NAME AND t.VERSION = s.VERSION")
                .whenNotMatchedInsertAll()
                .execute()
        )


    def run_add_table_for_model_info_1_first_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_record_model_module, "tasklist": self.ml_supplier_po_item_record_model_module},
        )


    def assertion_add_table_for_model_info_1_first_round(self):
        target_table_full_name = sbnutils.get_table_full_name(Zone.GENERAL.value, "ml_model_info")
        target_table_list = spark.sql(
                f"SELECT * FROM {target_table_full_name} t ORDER BY t.NAME"  ).collect()

        # sbnutils.log_info(target_table_list)

        # Check whether model_info has been correctly inserted or not
        assert len(target_table_list) == 3
        assert target_table_list[0]["NAME"] == self.model_name
        assert target_table_list[0]["VERSION"] == self.model_version
        assert target_table_list[0]["STAGE"] == self.model_stage

        
        assert target_table_list[1]["NAME"] == "zzztest_po_clf"
        assert target_table_list[1]["VERSION"] == 27
        assert target_table_list[1]["STAGE"] == "Staging"
        
        assert target_table_list[2]["NAME"] == "zzztest_po_reg"
        assert target_table_list[2]["VERSION"] == 28
        assert target_table_list[2]["STAGE"] == "Dev"
        self.assert_job_config_table(self.ml_supplier_po_item_record_model_module)


    def before_add_table_for_model_info_2_second_round(self):
        # Make sure that information of a seen model would not be updated or inserted

        # Get target_table
        target_zone = Zone.GENERAL.value
        target_table_name = "ml_model_info"
        target_table_full_name = sbnutils.get_table_full_name(target_zone, target_table_name)
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.NAME").collect()

        self.test_model_uuid = target_table_list[0]["UUID"]

    def run_add_table_for_model_info_2_second_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_record_model_module, "tasklist": self.ml_supplier_po_item_record_model_module},
        )

    def assertion_add_table_for_model_info_2_second_round(self):
        target_zone = Zone.GENERAL.value
        target_table_name = "ml_model_info"
        target_table_full_name = sbnutils.get_table_full_name(target_zone, target_table_name)
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.NAME").collect()

        # sbnutils.log_info(target_table_list)
        assert target_table_list[0]["UUID"] == self.test_model_uuid
        self.assert_job_config_table(self.ml_supplier_po_item_record_model_module)

# COMMAND ----------

result = PoItemRecordModelFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
