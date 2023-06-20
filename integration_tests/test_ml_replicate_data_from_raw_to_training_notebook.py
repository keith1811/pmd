# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - Replication job

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
from decimal import *
from integration_tests.quality_utils import SbnNutterFixture

from utils import *
from utils.constants import Zone, Table

from modules.utils.constants import MLTable

class ReplicationFixture(SbnNutterFixture):
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_replicate_data_from_raw_to_training_job"
        self.import_module = "ml_replicate_data_from_raw_to_training_module"
        self.test_update_time = datetime.datetime.now()
        self.BUYER_DATA_SCHEMA = 'ID Decimal(28,0), COMMODITY_ID Decimal(28,0), DESCRIPTION String, SUPPLIER_PART String, SUPPLIER_PART_EXTENSION String, MANUFACTURER_PART String, MANUFACTURER_NAME String, UNIT_OF_MEASURE String, ITEM_TYPE String, ITEM_CATEGORY String, BUYER_PART_ID String, TRANSPORT_TERMS String, DOMAIN String, CODE String, IS_ADHOC int, _ID Decimal(28,0), _CHANGED_DATETIM Timestamp, _DELTA_CREATED_ON Timestamp, _DELTA_UPDATED_ON Timestamp'
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql("training", f'{os.path.dirname(sqls.__file__)}/training')
        database_utils.execute_ddl_sql(Zone.RAW.value, f'{os.path.dirname(sqls.__file__)}/raw')
        database_utils.create_all_raw_tables()

    def before_ml_replicate_data_from_raw_to_training_1_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")

        # Mock data to test inserting
        timestamp_current = datetime.datetime.now()
        commodity_test_df = spark.createDataFrame([
            (Decimal(30000000000000000003852732), Decimal(30000000000000000007794776), "Laptop 4 PK", "cui-20230425_008_1","", "AX4518", "20008496", "PK", "Material",	"subcontract", "	selina-20230425_008_1", "",	"SPSC",	"0", 0, Decimal(30000000000000000003852732),  timestamp_current, timestamp_current, timestamp_current),
            (Decimal(30000000000000000003852733), Decimal(30000000000000000007794776), "Laptop 4 PK", "cui-20230425_008_1","", "AX4518", "20008496", "PK", "Material",	"subcontract", "	selina-20230425_008_1", "",	"SPSC",	"0", 0, Decimal(30000000000000000003852732),  timestamp_current, timestamp_current, timestamp_current),
            (Decimal(30000000000000000003852734), Decimal(30000000000000000007794776), "Laptop 4 PK", "cui-20230425_008_1","", "AX4518", "20008496", "PK", "Material",	"subcontract", "	selina-20230425_008_1", "",	"SPSC",	"0", 0, Decimal(30000000000000000003852732),  timestamp_current, timestamp_current, timestamp_current),
        ],schema=self.BUYER_DATA_SCHEMA)

        # Write mock data to the source table
        (
            sbnutils.get_delta_table(Zone.RAW.value, MLTable.RAW_ML_BUYER_PO_ITEM.value).alias("t")
            .merge(commodity_test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )
        

    def run_ml_replicate_data_from_raw_to_training_1_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.import_module, "tasklist": self.import_module},
        )

    def assertion_ml_replicate_data_from_raw_to_training_1_fisrt_round(self):
        # Get target table
        target_table_full_name = sbnutils.get_table_full_name(
            "training", MLTable.TRAINING_ML_BUYER_PO_ITEM.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID"
        ).collect()

        # Assert replication
        assert len(target_table_list) == 3

        # Assert job config table
        self.assert_job_config_table(self.import_module)

    def before_ml_replicate_data_from_raw_to_training_2_fisrt_round(self):
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")

        # Get current "_DELTA_UPDATED_ON" of the 1st data
        target_table_full_name = sbnutils.get_table_full_name("training", MLTable.TRAINING_ML_BUYER_PO_ITEM.value)
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID").collect()
        self.test_update_time = target_table_list[0]["_DELTA_UPDATED_ON"]

        # Add new data to test both inserting and updating
        timestamp_current = datetime.datetime.now()
        test_df = spark.createDataFrame([
            (Decimal(30000000000000000003852732), Decimal(30000000000000000007794776), "Laptop 4 PK", "cui-20230425_008_1","", "AX4518", "20008496", "PK", "Material",	"subcontract", "	selina-20230425_008_1", "",	"SPSC",	"0", 0, Decimal(30000000000000000003852732),  timestamp_current, timestamp_current, timestamp_current),
            (Decimal(30000000000000000003852735), Decimal(30000000000000000007794776), "Laptop 4 PK", "cui-20230425_008_1","", "AX4518", "20008496", "PK", "Material",	"subcontract", "	selina-20230425_008_1", "",	"SPSC",	"0", 0, Decimal(30000000000000000003852732),  timestamp_current, timestamp_current, timestamp_current)  
        ],schema=self.BUYER_DATA_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.RAW.value, MLTable.RAW_ML_BUYER_PO_ITEM.value).alias("t")
            .merge(test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )

    def run_ml_replicate_data_from_raw_to_training_2_fisrt_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.import_module, "tasklist": self.import_module},
        )

    def assertion_ml_replicate_data_from_raw_to_training_2_fisrt_round(self):
        # Get target table
        target_table_full_name = sbnutils.get_table_full_name(
            "training", MLTable.TRAINING_ML_BUYER_PO_ITEM.value
        )
        target_table_list = spark.sql(
            f"SELECT * FROM {target_table_full_name} t ORDER BY t.ID"
        ).collect()

        # Make sure that both inserting and updating work correctly
        assert target_table_list[0]["_DELTA_UPDATED_ON"] != self.test_update_time
        assert len(target_table_list) == 4

        # Assert job config table
        self.assert_job_config_table(self.import_module)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = ReplicationFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)

# COMMAND ----------


