# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - Sample batch job

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
from pyspark.sql.functions import current_timestamp

from utils import *
from utils.constants import Zone, Table

from modules.utils.constants import MLTable

# COMMAND ----------

class SampleBatchFixture(SbnNutterFixture):
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_eval_descr_job"
        self.ml_supplier_po_item_eval_descr_module = "ml_supplier_po_item_eval_descr_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')
        database_utils.execute_ddl_sql(Zone.RAW.value, f'{os.path.dirname(sqls.__file__)}/raw')
        database_utils.create_all_raw_tables()

    # 1. Test fisrt round
    def before_eval_descr_1_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        commodity_table_full_name = sbnutils.get_table_full_name(Zone.RAW.value, "commodity")
        # Add the test data to the source table
        timestamp_current = datetime.datetime.now()
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ("1001", 10001, 'ABCDEFG JIJISKEJO SASLJ', '11023012', timestamp_current, timestamp_current),
            ("1002", 10001, 'AB. CD', '11023000', timestamp_current, timestamp_current),
            ("1003", 10002, '  AB1234  ', '0', timestamp_current, timestamp_current),
            ("1004", 10003, 'ABCDEFG', '11022000', timestamp_current, timestamp_current),
            ("1005", 10004, None, None, timestamp_current, timestamp_current)
            ],schema='ID string, PO_ITEM_ID long, DESCRIPTION string, AN_UNSPSC_COMMODITY string, _DELTA_CREATED_ON timestamp, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
            .merge(ml_supplier_po_item_test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )
        commodity_test_df = spark.createDataFrame([
            (1001, 11023012, 1),
            (1002, 110230, 1),
            (1003, 110220, 0)
            ],schema='ID long, CODE long, VALID int')
        (
            sbnutils.get_delta_table(Zone.RAW.value, "commodity").alias("t")
            .merge(commodity_test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )

    def run_eval_descr_1_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_eval_descr_module, "tasklist": self.ml_supplier_po_item_eval_descr_module},
        )

    def assertion_eval_descr_1_fisrt_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(
            Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID"
        ).collect()
        sbnutils.log_info(target_table_list)
        assert len(target_table_list) == 5
        assert target_table_list[0]["ID"] == "1001"
        assert target_table_list[0]["PO_ITEM_ID"] == 10001
        assert target_table_list[0]["PROCESSED_DESCRIPTION"] == "abcdefg jijiskejo saslj"
        assert target_table_list[0]["AN_UNSPSC_SEGMENT"] == "11"
        assert target_table_list[0]["AN_UNSPSC_FAMILY"] == "1102"
        assert target_table_list[0]["AN_UNSPSC_CLASS"] == "110230"
        assert target_table_list[0]["AN_UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[0]['AN_CLASSIFICATION_QUALITY_COMMODITY'] == 'Acceptable'
        assert target_table_list[0]["AN_DATA_QUALITY_LEVEL"] == "Good"
        assert target_table_list[1]["ID"] == "1002"
        assert target_table_list[1]["PO_ITEM_ID"] == 10001
        assert target_table_list[1]["PROCESSED_DESCRIPTION"] == "ab  cd"
        assert target_table_list[1]["AN_UNSPSC_CLASS"] == "110230"
        assert target_table_list[1]["AN_UNSPSC_COMMODITY"] == "110230"
        assert target_table_list[1]['AN_CLASSIFICATION_QUALITY_COMMODITY'] == 'Acceptable'
        assert target_table_list[1]["AN_DATA_QUALITY_LEVEL"] == "Acceptable"
        assert target_table_list[2]["ID"] == "1003"
        assert target_table_list[2]["PO_ITEM_ID"] == 10002
        assert target_table_list[2]["PROCESSED_DESCRIPTION"] == "ab1234"
        assert target_table_list[2]["AN_UNSPSC_SEGMENT"] == "0"
        assert target_table_list[2]['AN_CLASSIFICATION_QUALITY_COMMODITY'] == 'Poor'
        assert target_table_list[2]["AN_DATA_QUALITY_LEVEL"] == "Poor"
        assert target_table_list[3]["ID"] == "1004"
        assert target_table_list[3]["PO_ITEM_ID"] == 10003
        assert target_table_list[3]["PROCESSED_DESCRIPTION"] == "abcdefg"
        assert target_table_list[3]["AN_UNSPSC_SEGMENT"] == "0"
        assert target_table_list[3]['AN_CLASSIFICATION_QUALITY_COMMODITY'] == 'Poor'
        assert target_table_list[3]["AN_DATA_QUALITY_LEVEL"] == "Acceptable"
        assert target_table_list[4]["ID"] == "1005"
        assert target_table_list[4]["PO_ITEM_ID"] == 10004
        assert target_table_list[4]["PROCESSED_DESCRIPTION"] == ""
        assert target_table_list[4]["AN_UNSPSC_SEGMENT"] == "0"
        assert target_table_list[4]['AN_CLASSIFICATION_QUALITY_COMMODITY'] == 'Poor'
        assert target_table_list[4]["AN_DATA_QUALITY_LEVEL"] == "Poor"

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_eval_descr_module)

    # 2. Test second round
    def before_eval_descr_2_second_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        commodity_table_full_name = sbnutils.get_table_full_name(Zone.RAW.value, "commodity")
        # Add the test data to the source table
        cur_timestamp = current_timestamp()
        target_columns = ['ID', 'PO_ITEM_ID', 'DESCRIPTION', 'AN_UNSPSC_COMMODITY']
        update_expr = {f"t.{c}": f"s.{c}" for c in target_columns}
        update_expr['_DELTA_UPDATED_ON'] = cur_timestamp

        insert_expr = update_expr.copy()
        insert_expr['_DELTA_CREATED_ON'] = cur_timestamp

        timestamp_current = datetime.datetime.now()
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ("1003", 10002, '  AB1234BB  ', '0', timestamp_current, timestamp_current),
            ("1004", 10003, 'ABCDEFG', '11023012', timestamp_current, timestamp_current),
            ("1006", 10005, 'ABCDEFG JIJISKEJO SASLJ', '11023012', timestamp_current, timestamp_current),
            ],schema='ID string, PO_ITEM_ID long, DESCRIPTION string, AN_UNSPSC_COMMODITY string, _DELTA_CREATED_ON timestamp, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
            .merge(ml_supplier_po_item_test_df.alias("s"), "t.ID = s.ID")
            .whenMatchedUpdate(set=update_expr)
            .whenNotMatchedInsert(values=insert_expr)
            .execute()
        )

    def run_eval_descr_2_second_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_eval_descr_module, "tasklist": self.ml_supplier_po_item_eval_descr_module},
        )

    def assertion_eval_descr_2_second_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(
            Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID"
        ).collect()
        sbnutils.log_info(target_table_list)
        assert len(target_table_list) == 6
        original_create_time = target_table_list[0]["_DELTA_CREATED_ON"]
        new_update_time = target_table_list[5]["_DELTA_UPDATED_ON"]
        assert target_table_list[0]["ID"] == "1001"
        assert target_table_list[0]["_DELTA_UPDATED_ON"] < new_update_time
        assert target_table_list[1]["ID"] == "1002"
        assert target_table_list[1]["_DELTA_UPDATED_ON"] < new_update_time
        assert target_table_list[2]["ID"] == "1003"
        assert target_table_list[2]["PROCESSED_DESCRIPTION"] == "ab1234bb"
        assert target_table_list[2]["_DELTA_UPDATED_ON"] == new_update_time
        assert target_table_list[2]["_DELTA_CREATED_ON"] == original_create_time
        assert target_table_list[3]["ID"] == "1004"
        assert target_table_list[3]["AN_UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[3]["_DELTA_UPDATED_ON"] == new_update_time
        assert target_table_list[3]["_DELTA_CREATED_ON"] == original_create_time
        assert target_table_list[4]["ID"] == "1005"
        assert target_table_list[4]["_DELTA_UPDATED_ON"] < new_update_time
        assert target_table_list[5]["ID"] == "1006"
        assert target_table_list[5]["PO_ITEM_ID"] == 10005
        assert target_table_list[5]["_DELTA_UPDATED_ON"] == new_update_time
        assert target_table_list[5]["_DELTA_CREATED_ON"] > original_create_time

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_eval_descr_module)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = SampleBatchFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
