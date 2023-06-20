# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - ML supplier po item load job

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import os
import sqls
import datetime
from integration_tests.quality_utils import SbnNutterFixture
from utils import *
from utils.constants import Zone, Table, DELTA_CREATED_FIELD, DELTA_UPDATED_FIELD
from modules.utils.constants import MLTable

# COMMAND ----------

class MlSupplierPoItemLoadBatchFixture(SbnNutterFixture):
    ML_SUPPLIER_PO_ITEM_SCHEMA = f"""
         ID STRING, PO_ITEM_ID LONG, DESCRIPTION STRING, DASHBOARD_STATUS STRING,
         FINAL_REPORT_UNSPSC_SEGMENT STRING,
         FINAL_REPORT_UNSPSC_FAMILY STRING,
         FINAL_REPORT_UNSPSC_CLASS STRING,
         FINAL_REPORT_UNSPSC_COMMODITY STRING,
         FINAL_REPORT_CONFIDENCE_SEGMENT Float,
         FINAL_REPORT_CONFIDENCE_FAMILY Float,
         FINAL_REPORT_CONFIDENCE_CLASS Float,
         FINAL_REPORT_CONFIDENCE_COMMODITY Float,
         FINAL_REPORT_LASTUPDATED_AT TIMESTAMP,
         FINAL_REPORT_SOURCE STRING,
        {DELTA_CREATED_FIELD} TIMESTAMP, {DELTA_UPDATED_FIELD} TIMESTAMP
    """

    FACT_SUPPLIER_PO_ITEM_SCHEMA = f"""
        ID LONG,
        PRODUCT_ID STRING,
        IS_BLANKET INT,
        DESCRIPTION STRING,
        {DELTA_CREATED_FIELD} TIMESTAMP, {DELTA_UPDATED_FIELD} TIMESTAMP
    """

    first_round_dim_table_product_id = None

    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_load_job"
        self.ml_supplier_po_item_load_module = "ml_supplier_po_item_load_module"
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        SbnNutterFixture.__init__(self, case_name)

        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.CONSUMPTION.value, f'{os.path.dirname(sqls.__file__)}/consumption')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')

    # 1. first round: first batch
    # Case 1: insert new data
    # ml_supplier_po_item
    # ID    PO_ITEM_ID  DASHBOARD_STATUS    FINAL_COMMODITY
    # 1     10001       PO_NEW              11111111
    # 2     10001       PO_NEW              11111111
    # 3     10001       PO_NEW              11111111

    # dim
    # ID    PO_ITEM_ID  FINAL_COMMODITY   createdon   updatedon
    # 1     10001       11111111

    # fact
    # ID    PRODUCT_ID  IS_BLANKET
    # 10001    1           0
    def before_ml_supplier_po_item_load_1_first_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)
        # insert ml_po_item table data
        ml_po_item_df = spark.createDataFrame([
            ('ID-10001-01', 10001, 'pid-10001-ml', 'PO_NEW', '11', '1111', '111111', '11111111',
             *(0.11111 for _ in range(4)), timestamp_earliest, 'gds', timestamp_earliest, timestamp_earliest),
            ('ID-10001-02', 10001, 'pid-10001-ml', 'PO_NEW', '11', '1111', '111111', '11111111',
             *(0.11111 for _ in range(4)), timestamp_earliest, 'gds', timestamp_earliest, timestamp_earliest)
        ], schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(ml_po_item_df.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert into fact_po_item
        fact_po_item_df = spark.createDataFrame([
            (10001, None, 0, 'pid-10001-fact', timestamp_earliest, timestamp_earliest)
        ], schema=self.FACT_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.CONSUMPTION.value, Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(fact_po_item_df.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_load_1_first_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_load_module, "tasklist": self.ml_supplier_po_item_load_module}
        )

    def assertion_ml_supplier_po_item_load_1_first_round(self):
        timestamp_current = datetime.datetime.now()

        target_dim_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value,
                                                                  MLTable.CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM.value)
        target_dim_table_list = spark.sql(
            f"SELECT * FROM {target_dim_table_full_name} t ORDER BY t.PO_ITEM_ID"
        ).collect()

        # 1. assert dimension table
        assert len(target_dim_table_list) == 1
        assert target_dim_table_list[0]["ID"].find("ID-10001-") != -1
        assert target_dim_table_list[0]["DESCRIPTION"] == "pid-10001-ml"
        assert target_dim_table_list[0]["UNSPSC_COMMODITY"] == "11111111"
        assert str(target_dim_table_list[0]["UNSPSC_CONFIDENCE_COMMODITY"]) == "0.11111"
        assert target_dim_table_list[0][DELTA_CREATED_FIELD] == target_dim_table_list[0][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_dim_table_list[0], timestamp_current)

        # store first round product id of 10001
        self.first_round_dim_table_product_id = target_dim_table_list[0]["ID"]
        sbnutils.log_info(f"First round dim table product id: {self.first_round_dim_table_product_id}")

        # 2. assert fact table
        target_fact_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value,
                                                                   Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value)
        target_fact_table_list = spark.sql(
            f"SELECT * FROM {target_fact_table_full_name} t WHERE PRODUCT_ID IS NOT NULL ORDER BY t.ID"
        ).collect()

        assert len(target_fact_table_list) == 1
        assert target_fact_table_list[0]["ID"] == 10001
        assert target_fact_table_list[0]["PRODUCT_ID"] == self.first_round_dim_table_product_id
        assert target_fact_table_list[0][DELTA_CREATED_FIELD] < target_fact_table_list[0][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_fact_table_list[0], timestamp_current)

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_load_module)

    # 2. second round:
    # Case 2: ID 1-3 update the final commodity, will not insert new columns, dbupdated will change.
    # ml_supplier_po_item
    # ID    PO_ITEM_ID  DASHBOARD_STATUS    FINAL_COMMODITY
    # 1     10001       PO_NEW              22222222
    # 2     10001       PO_NEW              22222222
    # 3     10001       PO_NEW              22222222

    # after filter out po_obsolete and drop duplicate, will fetch one record with updated final commodity
    # dim
    # ID    PO_ITEM_ID  FINAL_COMMODITY
    # 1     10001       11111111
    # 2     10001       22222222

    # merge condition: dim.PO_ITEM_ID == fact.ID and (fact.PRODUCT_ID is null or fact.PRODUCT_ID != dim.ID)
    # fact
    # ID    PRODUCT_ID
    # 10001 2
    def before_ml_supplier_po_item_load_2_second_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_current = datetime.datetime.now()
        timestamp_before_current = timestamp_current - datetime.timedelta(minutes=10)
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)

        ml_po_item_df = spark.createDataFrame([
            ('ID-10001-01', 10001, 'pid-10001-ml-case2', 'PO_NEW', '22', '2222', '222222', '22222222',
             *(0.22222 for _ in range(4)), timestamp_earliest, 'gds', timestamp_earliest, timestamp_before_current),
            ('ID-10001-02', 10001, 'pid-10001-ml-case2', 'PO_NEW', '22', '2222', '222222', '22222222',
             *(0.22222 for _ in range(4)), timestamp_earliest, 'gds', timestamp_earliest, timestamp_before_current)
        ], schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(ml_po_item_df.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # fix the fact table
        fact_po_item_df = spark.createDataFrame([
            (10001, self.first_round_dim_table_product_id, 0, 'pid-10001-fact', timestamp_earliest, timestamp_earliest)
        ], schema=self.FACT_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.CONSUMPTION.value, Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(fact_po_item_df.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_load_2_second_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_load_module, "tasklist": self.ml_supplier_po_item_load_module}
        )

    def assertion_ml_supplier_po_item_load_2_second_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_current = datetime.datetime.now()

        target_dim_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value,
                                                                  MLTable.CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM.value)
        target_dim_table_list = spark.sql(
            f"SELECT * FROM {target_dim_table_full_name} t ORDER BY t.PO_ITEM_ID, t.{DELTA_UPDATED_FIELD} ASC"
        ).collect()

        # 1. assert dim table
        sbnutils.log_info(f"Second case, dim table len: {len(target_dim_table_list)}")
        assert len(target_dim_table_list) <= 2
        if len(target_dim_table_list) == 1:
            # 1.1 dropDuplicate po_item_id is stable, will not insert new data into dim table
            assert target_dim_table_list[0]["ID"] == self.first_round_dim_table_product_id
            assert target_dim_table_list[0]["DESCRIPTION"] == "pid-10001-ml-case2"
            assert target_dim_table_list[0]["UNSPSC_COMMODITY"] == "22222222"
            assert target_dim_table_list[0][DELTA_CREATED_FIELD] < target_dim_table_list[0][DELTA_UPDATED_FIELD]
            self.assert_update_timestamp(target_dim_table_list[0], timestamp_current)

        else:
            # 1.2 dropDuplicate po_item_id becomes unstable, will insert a new data into dim table with the same po_item_id
            assert target_dim_table_list[0]["ID"] == self.first_round_dim_table_product_id
            assert target_dim_table_list[0]["DESCRIPTION"] == "pid-10001-ml"  # old data of 10001
            assert target_dim_table_list[0]["UNSPSC_COMMODITY"] == "11111111"
            assert target_dim_table_list[0][DELTA_CREATED_FIELD] == target_dim_table_list[0][DELTA_UPDATED_FIELD]
            assert target_dim_table_list[0][DELTA_UPDATED_FIELD] < target_dim_table_list[1][DELTA_UPDATED_FIELD]

            assert target_dim_table_list[1]["ID"].find("ID-10001-") != -1
            assert target_dim_table_list[1]["ID"] != self.first_round_dim_table_product_id
            assert target_dim_table_list[1]["DESCRIPTION"] == "pid-10001-ml-case2"  # new data of 10001
            assert target_dim_table_list[1]["UNSPSC_COMMODITY"] == "22222222"
            assert target_dim_table_list[1][DELTA_CREATED_FIELD] == target_dim_table_list[1][DELTA_UPDATED_FIELD]
            self.assert_update_timestamp(target_dim_table_list[1], timestamp_current)

        # 2. assert fact table
        target_fact_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value,
                                                                   Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value)
        target_fact_table_list = spark.sql(
            f"SELECT * FROM {target_fact_table_full_name} t WHERE PRODUCT_ID IS NOT NULL ORDER BY t.ID"
        ).collect()

        assert len(target_fact_table_list) == 1
        assert target_fact_table_list[0]["ID"] == 10001
        assert target_fact_table_list[0]["DESCRIPTION"] == "pid-10001-fact"
        if target_fact_table_list[0]["PRODUCT_ID"] == self.first_round_dim_table_product_id:
            assert target_fact_table_list[0][DELTA_UPDATED_FIELD] == timestamp_earliest
        else:
            self.assert_update_timestamp(target_fact_table_list[0], timestamp_current)

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_load_module)

    # case 3: id 1-2 obsolete, a new record will be inserted. Reprocessing
    # ID    PO_ITEM_ID  DASHBOARD_STATUS    FINAL_COMMODITY
    # 1     10001       PO_OBSOLETE         22222222
    # 2     10001       PO_OBSOLETE         22222222
    # 3     10002       PO_NEW              11111112
    # 4     10003       PO_NEW              11111112

    # filter out obsolete and drop duplicate, 10001 won't be updated. But a new record will be inserted.
    # dim
    # ID    PO_ITEM_ID  FINAL_COMMODITY
    # 1     10001       11111111   # won't be update
    # 3     10002       11111112
    # 4     10003       11111112

    # fact
    # ID    PRODUCT_ID    IS_BLANKET
    # 10001     1           0
    # 10002     3           0
    # 10003    None         1

    def before_ml_supplier_po_item_load_3_third_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_current = datetime.datetime.now()
        timestamp_before_current_ten_minutes = timestamp_current - datetime.timedelta(minutes=10)
        timestamp_before_current_ten_days = timestamp_current - datetime.timedelta(days=10)
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)

        ml_po_item_df = spark.createDataFrame([
            ('ID-10001-01', 10001, 'pid-10001-ml-case3', 'PO_OBSOLETE', '22', '2222', '222222', '22222222',
             *(0.22222 for _ in range(4)), timestamp_earliest, 'gds', timestamp_earliest, timestamp_current),
            # obsolete
            ('ID-10001-02', 10001, 'pid-10001-ml-case3', 'PO_OBSOLETE', '22', '2222', '222222', '22222222',
             *(0.22222 for _ in range(4)), timestamp_earliest, 'gds', timestamp_earliest, timestamp_current),
            # obsolete
            ('ID-10002-01', 10002, 'pid-10002-ml-case3', 'PO_NEW', '33', '3333', '3333', '33333333',
             *(0.33333 for _ in range(4)), timestamp_earliest, 'SBN', timestamp_earliest,
             timestamp_before_current_ten_minutes),
            ('ID-10003-01', 10003, 'pid-10003-ml-case3', 'PO_NEW', '33', '3333', '3333', '33333333',
             *(0.33333 for _ in range(4)), timestamp_earliest, 'SBN', timestamp_earliest,
             timestamp_before_current_ten_days),
            ('ID-10004-01', 10004, 'pid-10004-ml-case3', 'PO_NEW', '33', '3333', '3333', '33333333',
             *(0.33333 for _ in range(4)), timestamp_earliest, 'SBN', timestamp_earliest, timestamp_current),
            ('ID-10005-01', 10005, 'pid-10005-ml-case3', 'PO_NEW', '33', '3333', '3333', '33333333',
             *(0.33333 for _ in range(4)), timestamp_earliest, 'SBN', timestamp_earliest, timestamp_after_current),
            ('ID-10006-01', 10006, 'pid-10006-ml-case3', 'PO_NEW', '33', '3333', '3333', '33333333',
             *(0.33333 for _ in range(4)), timestamp_earliest, 'SBN', timestamp_earliest, timestamp_current),  # blanket
        ], schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(ml_po_item_df.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        # insert fact table
        fact_po_item_df = spark.createDataFrame([
            (10001, 'ID-10001-01', 0, 'pid-10001-fact-case3', timestamp_earliest, timestamp_earliest),
            # will not change
            (10002, None, 0, 'pid-10002-fact-case3', timestamp_earliest, timestamp_earliest),
            (10003, None, 0, 'pid-10003-fact-case3', timestamp_earliest, timestamp_earliest),
            (10004, None, 0, 'pid-10003-fact-case3', timestamp_earliest, timestamp_earliest),
            (10005, None, 0, 'pid-10003-fact-case3', timestamp_earliest, timestamp_earliest),
            (10006, None, 1, 'pid-10003-fact-case3', timestamp_earliest, timestamp_earliest)  # blanket not be updated
        ], schema=self.FACT_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.CONSUMPTION.value, Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(fact_po_item_df.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_load_3_third_round(self):
        # Update reset flag to get ready to reprocess data
        batch_utils.update_offset_timestamp_reset_flag(self.ml_supplier_po_item_load_module, 1)
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_load_module, "tasklist": self.ml_supplier_po_item_load_module}
        )

    def assertion_ml_supplier_po_item_load_3_third_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_current = datetime.datetime.now()

        target_dim_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value,
                                                                  MLTable.CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM.value)
        target_dim_table_list = spark.sql(
            f"SELECT * FROM {target_dim_table_full_name} t ORDER BY t.ID "
        ).collect()

        # 1. assert dim table
        assert 4 < len(target_dim_table_list) < 7  # 5 or 6
        assert target_dim_table_list[0]["ID"].find("ID-10001") != -1
        assert target_dim_table_list[0][DELTA_UPDATED_FIELD] < target_dim_table_list[-1][DELTA_UPDATED_FIELD]

        assert target_dim_table_list[-4]["ID"] == "ID-10002-01"
        self.assert_update_timestamp(target_dim_table_list[-4], timestamp_current)

        assert target_dim_table_list[-3]["ID"] == "ID-10003-01"
        self.assert_update_timestamp(target_dim_table_list[-3], timestamp_current)

        assert target_dim_table_list[-2]["ID"] == "ID-10004-01"
        assert target_dim_table_list[-2][DELTA_CREATED_FIELD] == target_dim_table_list[-2][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_dim_table_list[-2], timestamp_current)

        assert target_dim_table_list[-1]["ID"] == "ID-10006-01"
        assert target_dim_table_list[-1][DELTA_CREATED_FIELD] == target_dim_table_list[-1][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_dim_table_list[-1], timestamp_current)

        # 2. assert fact table
        target_fact_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value, Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value)
        target_fact_table_list = spark.sql(
            f"SELECT * FROM {target_fact_table_full_name} t WHERE PRODUCT_ID IS NOT NULL ORDER BY t.ID"
        ).collect()

        assert len(target_fact_table_list) == 4
        assert target_fact_table_list[0]["PRODUCT_ID"] == "ID-10001-01"  # will not change, because filter out obsolete 10001 data
        assert target_fact_table_list[0][DELTA_CREATED_FIELD] == timestamp_earliest
        assert target_fact_table_list[0][DELTA_UPDATED_FIELD] == timestamp_earliest

        assert target_fact_table_list[1]["PRODUCT_ID"] == "ID-10002-01"
        assert target_fact_table_list[1][DELTA_CREATED_FIELD] < target_fact_table_list[1][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_fact_table_list[1], timestamp_current)

        assert target_fact_table_list[2]["PRODUCT_ID"] == "ID-10003-01"
        assert target_fact_table_list[2][DELTA_CREATED_FIELD] < target_fact_table_list[2][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_fact_table_list[2], timestamp_current)

        assert target_fact_table_list[3]["PRODUCT_ID"] == "ID-10004-01"
        assert target_fact_table_list[3][DELTA_CREATED_FIELD] < target_fact_table_list[3][DELTA_UPDATED_FIELD]
        self.assert_update_timestamp(target_fact_table_list[3], timestamp_current)

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_load_module)

    def assert_update_timestamp(self, data, expected_timestamp):
        update_time_diff = abs(expected_timestamp - data[DELTA_UPDATED_FIELD]).total_seconds() / 60
        assert update_time_diff <= 60

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = MlSupplierPoItemLoadBatchFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)