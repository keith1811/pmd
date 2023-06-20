# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - Aggregate fact supplier po item batch job

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

class PoItemAggregationBatchFixture(SbnNutterFixture):
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_aggregate_batch_job"
        self.ml_supplier_po_item_aggregate_module = "ml_supplier_po_item_aggregate_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.CONSUMPTION.value, f'{os.path.dirname(sqls.__file__)}/consumption')

    def before_ml_supplier_po_item_aggregate_1_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        fact_supplier_po_item_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value, MLTable.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value)
        fact_network_po_item_table_full_name = sbnutils.get_table_full_name(Zone.CONSUMPTION.value, MLTable.CONSUMPTION_NETWORK_SUPPLIER_PO_ITEM.value)
        # Add the test data to the source table
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_earliest = timestamp_earliest - datetime.timedelta(
            microseconds=1
        )
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=10)
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ('100', 1001, 1001, 1001, 2001, datetime.datetime.strptime('2009-04-13 07:00:00', '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime('2023-03-27 15:13:42', '%Y-%m-%d %H:%M:%S'), 'China', 'EA', 10, 1000, "111", 0, timestamp_current, timestamp_current),
            ('100', 1001, 1002, 1001, 2001, datetime.datetime.strptime('2009-04-13 07:00:00', '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime('2023-03-27 15:13:42', '%Y-%m-%d %H:%M:%S'), 'China', 'EA', 15, 1200, "112", 0, timestamp_current, timestamp_current),
            ('101', 1002, 1003, 1001, 2002, datetime.datetime.strptime('2009-05-13 07:00:00', '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime('2023-03-27 15:13:42', '%Y-%m-%d %H:%M:%S'), 'China', 'EA', 10, 1000, "113", 0, timestamp_current, timestamp_current),
            ('101', 1002, 1004, 1001, 2002, datetime.datetime.strptime('2009-05-13 07:00:00', '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime('2023-03-27 15:13:42', '%Y-%m-%d %H:%M:%S'), 'China', 'EA', 10, 1000, "114", 0, timestamp_current, timestamp_current),
            ('101', 1005, 1005, 1001, 2002, datetime.datetime.strptime('2009-05-13 07:00:00', '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime('2023-03-27 15:13:42', '%Y-%m-%d %H:%M:%S'), 'USA', 'EA', 10, 1000, "115", 0, timestamp_current, timestamp_current),
            ('101', 1006, 1006, 1001, 2002, datetime.datetime.strptime('2009-05-13 07:00:00', '%Y-%m-%d %H:%M:%S'), datetime.datetime.strptime('2023-03-27 15:13:42', '%Y-%m-%d %H:%M:%S'), 'USA', 'KJ', 10, 1000, "116", 0, timestamp_current, timestamp_current)
        ],schema='SUPPLIER_ANID string, PO long, ID long, BUYER_ORG long, SUPPLIER_ORG long, REQUESTED_DELIVERY_DATE timestamp, DOCUMENT_DATE timestamp, SHIP_TO_COUNTRY string, UNIT_OF_MEASURE string, QUANTITY long, UNIT_PRICE_USD long, PRODUCT_ID string, IS_BLANKET int, _DELTA_CREATED_ON timestamp, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.CONSUMPTION.value, MLTable.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value).alias("t")
            .merge(ml_supplier_po_item_test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )
        dim_ml_supplier_po_item_df = spark.createDataFrame([
            ("111", '11', '1102', '110230', '11023012', timestamp_current),
            ("112", '11', '1102', '110230', '11023012', timestamp_current),
            ("113", '11', '1102', '110230', '11023012', timestamp_current),
            ("114", '11', '1102', '110230', '11023012', timestamp_current),
            ("115", '11', '1102', '110230', '11023012', timestamp_current),
            ("116", None, None, None, None, timestamp_current)
        ],schema = 'ID string, UNSPSC_SEGMENT string, UNSPSC_FAMILY string, UNSPSC_CLASS string, UNSPSC_COMMODITY string, _DELTA_UPDATED_ON timestamp')
        (
            sbnutils.get_delta_table(Zone.CONSUMPTION.value, MLTable.CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM.value).alias("t")
            .merge(dim_ml_supplier_po_item_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )


    def run_ml_supplier_po_item_aggregate_1_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_aggregate_module, "tasklist": self.ml_supplier_po_item_aggregate_module},
        )

    def assertion_ml_supplier_po_item_aggregate_1_fisrt_round(self):
        # Assert target table
        target_table_full_name = sbnutils.get_table_full_name(
            Zone.CONSUMPTION.value, MLTable.CONSUMPTION_NETWORK_SUPPLIER_PO_ITEM.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.SUPPLIER_ANID"
        ).collect()
        assert len(target_table_list) == 4
        assert target_table_list[0]["SUPPLIER_ANID"] == '100'
        assert target_table_list[0]["BUYER_ORG"] == 1001
        assert target_table_list[0]["SUPPLIER_ORG"] == 2001
        assert target_table_list[0]["REQUESTED_DELIVERY_DATE"] == datetime.datetime.strptime('2009-04-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[0]["DOCUMENT_DATE"] == datetime.datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[0]["UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[0]['SHIP_TO_COUNTRY'] == 'China'
        assert target_table_list[0]["UNIT_OF_MEASURE"] == "EA"
        assert target_table_list[0]["ORDER_AMOUNT"] == 28000
        assert target_table_list[0]["ORDER_UNITS"] == 25
        assert target_table_list[0]["PO_ITEM_COUNT"] == 2
        assert target_table_list[0]["AVERAGE_UNIT_PRICE_USD"] == 1100
        assert target_table_list[1]["SUPPLIER_ANID"] == '101'
        assert target_table_list[1]["BUYER_ORG"] == 1001
        assert target_table_list[1]["SUPPLIER_ORG"] == 2002
        assert target_table_list[1]["REQUESTED_DELIVERY_DATE"] == datetime.datetime.strptime('2009-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[1]["DOCUMENT_DATE"] == datetime.datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[1]["UNSPSC_COMMODITY"] == "99999999"
        assert target_table_list[1]['SHIP_TO_COUNTRY'] == 'USA'
        assert target_table_list[1]["UNIT_OF_MEASURE"] == "KJ"
        assert target_table_list[1]["ORDER_AMOUNT"] == 10000
        assert target_table_list[1]["ORDER_UNITS"] == 10
        assert target_table_list[1]["AVERAGE_UNIT_PRICE_USD"] == 1000
        assert target_table_list[1]["PO_ITEM_COUNT"] == 1
        assert target_table_list[2]["SUPPLIER_ANID"] == '101'
        assert target_table_list[2]["BUYER_ORG"] == 1001
        assert target_table_list[2]["SUPPLIER_ORG"] == 2002
        assert target_table_list[2]["REQUESTED_DELIVERY_DATE"] == datetime.datetime.strptime('2009-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[2]["DOCUMENT_DATE"] == datetime.datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[2]["UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[2]['SHIP_TO_COUNTRY'] == 'China'
        assert target_table_list[2]["UNIT_OF_MEASURE"] == "EA"
        assert target_table_list[2]["ORDER_AMOUNT"] == 20000
        assert target_table_list[2]["ORDER_UNITS"] == 20
        assert target_table_list[2]["AVERAGE_UNIT_PRICE_USD"] == 1000
        assert target_table_list[2]["PO_ITEM_COUNT"] == 2
        assert target_table_list[3]["SUPPLIER_ANID"] == '101'
        assert target_table_list[3]["BUYER_ORG"] == 1001
        assert target_table_list[3]["SUPPLIER_ORG"] == 2002
        assert target_table_list[3]["REQUESTED_DELIVERY_DATE"] == datetime.datetime.strptime('2009-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[3]["DOCUMENT_DATE"] == datetime.datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
        assert target_table_list[3]["UNSPSC_COMMODITY"] == "11023012"
        assert target_table_list[3]['SHIP_TO_COUNTRY'] == 'USA'
        assert target_table_list[3]["UNIT_OF_MEASURE"] == "EA"
        assert target_table_list[3]["ORDER_AMOUNT"] == 10000
        assert target_table_list[3]["ORDER_UNITS"] == 10
        assert target_table_list[3]["AVERAGE_UNIT_PRICE_USD"] == 1000
        assert target_table_list[3]["PO_ITEM_COUNT"] == 1

    


        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_aggregate_module)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = PoItemAggregationBatchFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
