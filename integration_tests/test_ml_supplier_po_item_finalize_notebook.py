# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - PO Item Finalize batch job

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
from decimal import *

from modules.utils.constants import MLTable

# COMMAND ----------

class PoItemFinalizeBatchFixture(SbnNutterFixture):
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_finalize_job"
        self.ml_supplier_po_item_finalize_module = "ml_supplier_po_item_finalize_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')
        database_utils.create_all_raw_tables()

    # 1. Test fisrt round
    def before_1_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        commodity_table_full_name = sbnutils.get_table_full_name(Zone.RAW.value, "commodity")
        # Add the test data to the source table
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_earliest = timestamp_earliest - datetime.timedelta(
            microseconds=1
        )
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ("1001", 10001, '11023010', '11023012', Decimal('0.85'), '11033012', Decimal('0.65'), '', timestamp_current, timestamp_current),
            ("1002", 10002, '11023010', '11023012', Decimal('0.75'), '11033012', Decimal('0.84'), '', timestamp_current, timestamp_current),
            ("1003", 10003, '11023012', '11023012', Decimal('0.65'), '11033012', Decimal('0.55'), '', timestamp_current, timestamp_current),
            ("1004", 10004, '11023010', '11023012', Decimal('0.65'), '11033012', Decimal('0.65'), '', timestamp_current, timestamp_current),
            ("1005", 10005, '11023010', '11023012', Decimal('0.65'), '11033012', Decimal('0.65'), 'Good', timestamp_current, timestamp_current)
            ],schema='ID string, PO_ITEM_ID long, AN_UNSPSC_COMMODITY String, EXTERNAL_PREDICATED_UNSPSC_COMMODITY String, EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY Decimal(6,5), SBN_PREDICATED_UNSPSC_COMMODITY String, SBN_PREDICTION_CONFIDENCE_COMMODITY Decimal(6,5), AN_CLASSIFICATION_QUALITY_COMMODITY String, _DELTA_CREATED_ON timestamp, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
            .merge(ml_supplier_po_item_test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )


    def run_1_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_finalize_module, "tasklist": self.ml_supplier_po_item_finalize_module},
        )

    def assertion_1_fisrt_round(self):
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
        assert target_table_list[0]["FINAL_REPORT_UNSPSC_SEGMENT"] == "11"
        assert target_table_list[0]["FINAL_REPORT_UNSPSC_FAMILY"] == "1102"
        assert target_table_list[0]["FINAL_REPORT_UNSPSC_CLASS"] == "110230"
        assert target_table_list[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[0]['FINAL_REPORT_CONFIDENCE_SEGMENT'] == Decimal('0.85')
        assert target_table_list[0]['FINAL_REPORT_CONFIDENCE_COMMODITY'] == Decimal('0.85')
        assert target_table_list[1]["ID"] == "1002"
        assert target_table_list[1]["PO_ITEM_ID"] == 10002
        assert target_table_list[1]["FINAL_REPORT_UNSPSC_SEGMENT"] == "11"
        assert target_table_list[1]["FINAL_REPORT_UNSPSC_FAMILY"] == "1103"
        assert target_table_list[1]["FINAL_REPORT_UNSPSC_CLASS"] == "110330"
        assert target_table_list[1]["FINAL_REPORT_UNSPSC_COMMODITY"] == "11033012"
        assert target_table_list[1]['FINAL_REPORT_CONFIDENCE_SEGMENT'] == Decimal('0.84')
        assert target_table_list[1]['FINAL_REPORT_CONFIDENCE_COMMODITY'] == Decimal('0.84')
        assert target_table_list[2]["ID"] == "1003"
        assert target_table_list[2]["PO_ITEM_ID"] == 10003
        assert target_table_list[2]["FINAL_REPORT_UNSPSC_SEGMENT"] == "11"
        assert target_table_list[2]["FINAL_REPORT_UNSPSC_FAMILY"] == "1102"
        assert target_table_list[2]["FINAL_REPORT_UNSPSC_CLASS"] == "110230"
        assert target_table_list[2]["FINAL_REPORT_UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[2]['FINAL_REPORT_CONFIDENCE_SEGMENT'] == Decimal('0.65')
        assert target_table_list[2]['FINAL_REPORT_CONFIDENCE_COMMODITY'] == Decimal('0.65')
        assert target_table_list[3]["ID"] == "1004"
        assert target_table_list[3]["PO_ITEM_ID"] == 10004
        assert target_table_list[3]["FINAL_REPORT_UNSPSC_SEGMENT"] is None
        assert target_table_list[3]["FINAL_REPORT_UNSPSC_FAMILY"] is None
        assert target_table_list[3]["FINAL_REPORT_UNSPSC_CLASS"] is None
        assert target_table_list[3]["FINAL_REPORT_UNSPSC_COMMODITY"] is None
        assert target_table_list[3]['FINAL_REPORT_CONFIDENCE_SEGMENT'] is None
        assert target_table_list[3]['FINAL_REPORT_CONFIDENCE_COMMODITY'] is None
        assert target_table_list[4]["ID"] == "1005"
        assert target_table_list[4]["PO_ITEM_ID"] == 10005
        assert target_table_list[4]["FINAL_REPORT_UNSPSC_SEGMENT"] == "11"
        assert target_table_list[4]["FINAL_REPORT_UNSPSC_FAMILY"] == "1102"
        assert target_table_list[4]["FINAL_REPORT_UNSPSC_CLASS"] == "110230"
        assert target_table_list[4]["FINAL_REPORT_UNSPSC_COMMODITY"] == "11023010"
        assert target_table_list[4]['FINAL_REPORT_CONFIDENCE_SEGMENT'] is None
        assert target_table_list[4]['FINAL_REPORT_CONFIDENCE_COMMODITY'] is None

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_finalize_module)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = PoItemFinalizeBatchFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
