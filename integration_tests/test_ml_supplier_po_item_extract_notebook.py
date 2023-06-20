# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - ML supplier po item batch job

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
from utils.constants import Zone, Table, DELTA_CREATED_FIELD, DELTA_UPDATED_FIELD

from modules.utils.constants import MLTable

# COMMAND ----------

class MlSupplierPoItemBatchFixture(SbnNutterFixture):
    PO_SCHEMA = """
        ID LONG, GENERIC_DOCUMENT LONG, IS_BLANKET INT, IS_ADHOC INT,
        _ID LONG, _CHANGED_DATETIME TIMESTAMP, _DELTA_CREATED_ON TIMESTAMP, _DELTA_UPDATED_ON TIMESTAMP 
    """

    CXML_DOCUMENT_SCHEMA = """
        ID LONG, FROM_ORG LONG, TO_ORG LONG, DASHBOARD_STATUS STRING,
        _ID LONG, _CHANGED_DATETIME TIMESTAMP, _DELTA_CREATED_ON TIMESTAMP, _DELTA_UPDATED_ON TIMESTAMP
    """

    PO_ITEM_SCHEMA = """
        ID LONG, PO LONG, DESCRIPTION STRING,
        _ID LONG, _CHANGED_DATETIME TIMESTAMP, _DELTA_CREATED_ON TIMESTAMP, _DELTA_UPDATED_ON TIMESTAMP
    """

    PO_ITEM_COMMODITY_SCHEMA = """
        ID LONG, PO_ITEM LONG, DOMAIN STRING, CODE STRING,
        _ID LONG, _CHANGED_DATETIME TIMESTAMP, _DELTA_CREATED_ON TIMESTAMP, _DELTA_UPDATED_ON TIMESTAMP
    """

    ORG_SCHEMA = """
        ID LONG, TYPE LONG, EFFECTIVENETWORK LONG, ANID STRING, NAME STRING, STATUS STRING,
        _ID LONG, _CHANGED_DATETIME TIMESTAMP, _DELTA_CREATED_ON TIMESTAMP, _DELTA_UPDATED_ON TIMESTAMP
    """

    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_extract_job"
        self.ml_supplier_po_item_extract_module = "ml_supplier_po_item_extract_module"
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.RAW.value, f'{os.path.dirname(sqls.__file__)}/raw')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')
        database_utils.create_all_raw_tables()

    # 1. first round: first batch
    def before_ml_supplier_po_item_enrichment_1_first_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_earliest = timestamp_earliest - datetime.timedelta(microseconds=1)
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)

        # insert test data into po
        df_po =  spark.createDataFrame([
            (10001, 20001, 0, 0, 10001, timestamp_current, timestamp_before_earliest, timestamp_before_earliest),
            (10002, 20002, 0, 0, 10002, timestamp_current, timestamp_earliest, timestamp_earliest),
            (10003, 20003, 0, 0, 10003, timestamp_current, timestamp_earliest, timestamp_current),
            (10004, 20004, 1, 0, 10004, timestamp_current, timestamp_earliest, timestamp_current),
            (10005, 20005, 0, 0, 10005, timestamp_current, timestamp_earliest, timestamp_after_current)

        ], schema = self.PO_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_PO.value).alias("t")
                .merge(df_po.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert test data into cxml_document
        df_cxml_document = spark.createDataFrame([
            (20001, 10000000, 20000000, 'PO_NEW', 10001, timestamp_current, timestamp_earliest, timestamp_current),
            (20002, 10000000, 20000000, 'PO_NEW', 10002, timestamp_current, timestamp_earliest, timestamp_current),
            (20003, 30000000, 40000000, 'PO_OBSOLETED', 10003, timestamp_current, timestamp_earliest, timestamp_current),
            (20004, 10000000, 20000000, 'PO_NEW', 10004, timestamp_current, timestamp_earliest, timestamp_current),
            (20005, 10000000, 20000000, 'PO_NEW', 10005, timestamp_current, timestamp_earliest, timestamp_current)
        ], schema = self.CXML_DOCUMENT_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_CXML_DOCUMENT.value).alias("t")
                .merge(df_cxml_document.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert test data into po_item
        df_po_item = spark.createDataFrame([
            (30001, 10001, 'po_item_1_1_description', 10001, timestamp_current, timestamp_earliest, timestamp_current),
            (30002, 10002, 'po_item_2_1_description', 10002, timestamp_current, timestamp_earliest, timestamp_current),
            (30003, 10002, 'po_item_2_2_description', 10002, timestamp_current, timestamp_earliest, timestamp_current),
            (30004, 10003, 'po_item_3_1_description', 10003, timestamp_current, timestamp_earliest, timestamp_current),
            (30005, 10003, 'po_item_3_2_description', 10003, timestamp_current, timestamp_earliest, timestamp_current),
            (30006, 10004, 'po_item_4_1_description', 10004, timestamp_current, timestamp_earliest, timestamp_current),
            (30007, 10005, 'po_item_5_1_description', 10005, timestamp_current, timestamp_earliest, timestamp_current)
        ], schema = self.PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_PO_ITEM.value).alias("t")
                .merge(df_po_item.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert test data into po_item_commodity
        df_po_item_commodity = spark.createDataFrame([
            (40001, 30001, 'unspsc', '11111111', 10001, timestamp_current, timestamp_earliest, timestamp_current),
            (40002, 30002, 'UNSPSC', '22222221', 10002, timestamp_current, timestamp_earliest, timestamp_current),
            (40003, 30002, 'testcc', '22222222', 10003, timestamp_current, timestamp_earliest, timestamp_current),
            (40004, 30003, 'UNSPSC', '33333331', 10004, timestamp_current, timestamp_earliest, timestamp_current),
            (40005, 30003, 'UNSPSC', '33333332', 10005, timestamp_current, timestamp_earliest, timestamp_current),
            (40006, 30004, 'UNSPSC', '44444441', 10006, timestamp_current, timestamp_earliest, timestamp_current)
        ], schema = self.PO_ITEM_COMMODITY_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_PO_ITEM_COMMODITY.value).alias("t")
                .merge(df_po_item_commodity.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        df_org = spark.createDataFrame([
            (10000000, 1000, 1000, '80000001', 'test_org_1', 'test_status', 10001, timestamp_current, timestamp_earliest, timestamp_current),
            (20000000, 1000, 1000, '80000002', 'test_org_2', 'test_status', 10002, timestamp_current, timestamp_earliest, timestamp_current),
            (30000000, 1000, 1000, '80000003', 'test_org_3', 'test_status', 10003, timestamp_current, timestamp_earliest, timestamp_current),
            (40000000, 1000, 1000, '80000004', 'test_org_4', 'test_status', 10004, timestamp_current, timestamp_earliest, timestamp_current)
        ], schema = self.ORG_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_ORG.value).alias("t")
                .merge(df_org.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_enrichment_1_first_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_extract_module, "tasklist": self.ml_supplier_po_item_extract_module},
        )

    def assertion_ml_supplier_po_item_enrichment_1_first_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        target_table_list = spark.sql(
                f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID ASC"
            ).collect()
        timestamp_current = datetime.datetime.now()

        assert len(target_table_list) == 6
        assert target_table_list[0]["ID"] == "30002-40002"
        assert target_table_list[0]["PO_ITEM_ID"] == 30002
        assert target_table_list[0]["AN_UNSPSC_COMMODITY"] == "22222221"
        assert target_table_list[0]["DASHBOARD_STATUS"] == "PO_NEW"

        assert target_table_list[1]["ID"] == "30002-40003"
        assert target_table_list[1]["PO_ITEM_ID"] == 30002
        assert target_table_list[1]["AN_UNSPSC_COMMODITY"] == "22222222"
        assert target_table_list[1]["BUYER_NAME"] == "test_org_1"
        assert target_table_list[1]["SUPPLIER_NAME"] == "test_org_2"

        assert target_table_list[2]["ID"] == "30003-40004"
        assert target_table_list[2]["PO_ITEM_ID"] == 30003
        assert target_table_list[2]["AN_UNSPSC_COMMODITY"] == "33333331"
        assert target_table_list[2]["DASHBOARD_STATUS"] == "PO_NEW"

        assert target_table_list[3]["ID"] == "30003-40005"
        assert target_table_list[3]["PO_ITEM_ID"] == 30003
        assert target_table_list[3]["AN_UNSPSC_COMMODITY"] == "33333332"
        assert target_table_list[3]["DASHBOARD_STATUS"] == "PO_NEW"

        assert target_table_list[4]["ID"] == "30004-40006"
        assert target_table_list[4]["PO_ITEM_ID"] == 30004
        assert target_table_list[4]["DASHBOARD_STATUS"] == "PO_OBSOLETED"
        assert target_table_list[4]["AN_UNSPSC_COMMODITY"] == "44444441"

        assert target_table_list[5]["ID"] == "30005-0"
        assert target_table_list[5]["PO_ITEM_ID"] == 30005
        assert target_table_list[5]["AN_UNSPSC_COMMODITY"] == None

        self.assert_timestamp(target_table_list[5], timestamp_current)

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_extract_module)

    # 2. second round: insert new data and update old data
    def before_ml_supplier_po_item_enrichment_2_second_round(self):
        # define timestamp
        timestamp_current = datetime.datetime.now()
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_current = timestamp_current - datetime.timedelta(minutes=10)
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)

        # insert test data into po
        df_po = spark.createDataFrame([
            (10002, 20002, 0, 0, 10002, timestamp_current, timestamp_earliest, timestamp_current),  # old data update
            (10017, 20017, 0, 0, 10017, timestamp_current, timestamp_earliest, timestamp_before_current),
            (10018, 20018, 0, 0, 10018, timestamp_current, timestamp_earliest, timestamp_current), # new data, update origin 20002
            (10019, 20019, 0, 0, 10019, timestamp_current, timestamp_earliest, timestamp_after_current)
        ], schema=self.PO_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_PO.value).alias("t")
                .merge(df_po.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert test data into cxml_document
        df_cxml_document = spark.createDataFrame([
            (20002, 10000000, 20000000, 'PO_UPDATED', 10002, timestamp_current, timestamp_earliest, timestamp_current),  # old data change to po_obsolete
            (20017, 10000000, 20000000, 'PO_NEW', 10017, timestamp_current, timestamp_earliest, timestamp_before_current),
            (20018, 10000000, 20000000, 'PO_NEW', 10018, timestamp_current, timestamp_earliest, timestamp_current), # new data
            (20019, 10000000, 20000000, 'PO_NEW', 10019, timestamp_current, timestamp_earliest, timestamp_current)
        ], schema=self.CXML_DOCUMENT_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_CXML_DOCUMENT.value).alias("t")
                .merge(df_cxml_document.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert new test data into po_item
        df_po_item = spark.createDataFrame([
            (30017, 10017, 'po_item_7_1_description', 10017, timestamp_current, timestamp_earliest, timestamp_before_current),
            (30018, 10018, 'po_item_8_1_description', 10018, timestamp_current, timestamp_earliest, timestamp_current), # new data
            (30019, 10019, 'po_item_9_1_description', 10019, timestamp_current, timestamp_earliest, timestamp_after_current)

        ], schema=self.PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_PO_ITEM.value).alias("t")
                .merge(df_po_item.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert test data into po_item_commodity
        df_po_item_commodity = spark.createDataFrame([
            (40017, 30017, 'UNSPSC', '77777777', 10017, timestamp_current, timestamp_earliest, timestamp_before_current),
            (40018, 30018, 'unspsc', '10', 10018, timestamp_current, timestamp_earliest, timestamp_current), # new data
            (40019, 30019, 'UNSPSC', '20', 10019, timestamp_current, timestamp_earliest, timestamp_after_current)
        ], schema=self.PO_ITEM_COMMODITY_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, Table.RAW_PO_ITEM_COMMODITY.value).alias("t")
                .merge(df_po_item_commodity.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )


    def run_ml_supplier_po_item_enrichment_2_second_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_extract_module, "tasklist": self.ml_supplier_po_item_extract_module},
        )

    def assertion_ml_supplier_po_item_enrichment_2_second_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        target_table_list = spark.sql(
                f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID ASC"
            ).collect()
        timestamp_current = datetime.datetime.now()

        assert len(target_table_list) == 7
        assert target_table_list[0]["ID"] == "30002-40002"
        assert target_table_list[0]["DASHBOARD_STATUS"] == "PO_UPDATED"
        assert target_table_list[0]["AN_UNSPSC_COMMODITY"] == "22222221"
        assert target_table_list[0]["_DELTA_CREATED_ON"] < target_table_list[0]["_DELTA_UPDATED_ON"]

        assert target_table_list[1]["ID"] == "30002-40003"
        assert target_table_list[1]["DASHBOARD_STATUS"] == "PO_UPDATED"
        assert target_table_list[1]["AN_UNSPSC_COMMODITY"] == "22222222"
        assert target_table_list[1]["_DELTA_CREATED_ON"] < target_table_list[1]["_DELTA_UPDATED_ON"]

        assert target_table_list[2]["ID"] == "30003-40004"
        assert target_table_list[2]["DASHBOARD_STATUS"] == "PO_UPDATED"
        assert target_table_list[2]["AN_UNSPSC_COMMODITY"] == "33333331"
        assert target_table_list[2]["_DELTA_CREATED_ON"] < target_table_list[1]["_DELTA_UPDATED_ON"]

        assert target_table_list[3]["ID"] == "30003-40005"
        assert target_table_list[3]["DASHBOARD_STATUS"] == "PO_UPDATED"
        assert target_table_list[3]["AN_UNSPSC_COMMODITY"] == "33333332"
        assert target_table_list[3]["_DELTA_CREATED_ON"] < target_table_list[1]["_DELTA_UPDATED_ON"]

        assert target_table_list[-1]["ID"] == "30018-40018"
        assert target_table_list[-1]["PO_ITEM_ID"] == 30018
        assert target_table_list[-1]["AN_UNSPSC_COMMODITY"] == "10"
        assert target_table_list[-1]["BUYER_ORG"] == 10000000
        assert target_table_list[-1]["SUPPLIER_ORG"] == 20000000

        self.assert_timestamp(target_table_list[-1], timestamp_current)

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_extract_module)

    # 5. fifth round: reprocessing
    def before_ml_supplier_po_item_enrichment_3_fifth_round(self):
        pass

    def run_ml_supplier_po_item_enrichment_3_fifth_round(self):
        # Update reset flag to get ready to reprocess data
        batch_utils.update_offset_timestamp_reset_flag(self.ml_supplier_po_item_extract_module, 1)
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_extract_module, "tasklist": self.ml_supplier_po_item_extract_module},
        )

    def assertion_ml_supplier_po_item_enrichment_3_fifth_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID ASC"
        ).collect()
        assert len(target_table_list) == 8
        assert target_table_list[-2]["ID"] == "30017-40017"
        assert target_table_list[-2]["PO_ITEM_ID"] == 30017
        assert target_table_list[-2]["AN_UNSPSC_COMMODITY"] == "77777777"

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_extract_module)

    def assert_timestamp(self, data, expected_timestamp):
        create_time_diff = abs(expected_timestamp - data[DELTA_CREATED_FIELD]).total_seconds() / 60
        update_time_diff = abs(expected_timestamp - data[DELTA_UPDATED_FIELD]).total_seconds() / 60
        assert create_time_diff <= 60
        assert update_time_diff <= 60

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = MlSupplierPoItemBatchFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)

# COMMAND ----------


