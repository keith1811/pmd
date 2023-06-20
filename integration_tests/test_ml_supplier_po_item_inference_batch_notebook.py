# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - ML supplier po item inference batch job

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
!cp ../requirements-ml.txt ~/.
%pip install -r ~/requirements-ml.txt
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import sys
import os
from modules.utils.feature_transform_utils import add_concat_feature_column_to_df
from modules.utils.spark_utils import replace_columns_values
import sqls
import datetime
from pyspark.sql import functions as f
from integration_tests.quality_utils import SbnNutterFixture
from utils.constants import Zone, Table, DELTA_CREATED_FIELD, DELTA_UPDATED_FIELD
import utils.sbnutils as sbnutils
import utils.database_utils as database_utils
import utils.batch_utils as batch_utils
from modules.utils.constants import MLTable
from modules.utils.config_utils import *

# COMMAND ----------

class MlSupplierPoItemInferenceBatchFixture(SbnNutterFixture):
    ML_MODEL_INFO_SCHEMA = f"""
    UUID STRING, NAME STRING, VERSION INT, STAGE STRING,
    {DELTA_CREATED_FIELD} TIMESTAMP, {DELTA_UPDATED_FIELD} TIMESTAMP
    """

    BNA_REPORT_SCHEMA = f"""
     CONCAT_FEATURE String,
     UNSPSC_SEGMENT String,
     UNSPSC_FAMILY String,
     UNSPSC_CLASS String,
     UNSPSC_COMMODITY String,
     CONFIDENCE_SEGMENT Decimal(6,5),
     CONFIDENCE_FAMILY Decimal(6,5),
     CONFIDENCE_CLASS Decimal(6,5),
     CONFIDENCE_COMMODITY Decimal(6,5),
     REPORT_LASTUPDATED_AT Timestamp,
     MODEL_UUID String,
     _DELTA_CREATED_ON Timestamp,
     _DELTA_UPDATED_ON Timestamp
    """

    ML_SUPPLIER_PO_ITEM_SCHEMA = f"""
    ID STRING, PO_ITEM_ID LONG, DESCRIPTION STRING, IS_ADHOC INT, BUYER_ORG LONG, SUPPLIER_ORG LONG, 
    SUPPLIER_PART STRING, MANUFACTURER_PART STRING, MANUFACTURER_NAME STRING,
    ITEM_TYPE STRING, 
     SBN_PREDICATED_UNSPSC_SEGMENT STRING,
     SBN_PREDICATED_UNSPSC_FAMILY STRING,
     SBN_PREDICATED_UNSPSC_CLASS STRING,
     SBN_PREDICATED_UNSPSC_COMMODITY STRING,
     SBN_PREDICTION_CONFIDENCE_SEGMENT DECIMAL(6,5),
     SBN_PREDICTION_CONFIDENCE_FAMILY DECIMAL(6,5),
     SBN_PREDICTION_CONFIDENCE_CLASS DECIMAL(6,5),
     SBN_PREDICTION_CONFIDENCE_COMMODITY DECIMAL(6,5),
     SBN_PREDICTION_LASTUPDATED_AT TIMESTAMP,
     MODEL_UUID STRING,
    {DELTA_CREATED_FIELD} TIMESTAMP, {DELTA_UPDATED_FIELD} TIMESTAMP
    """

    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_inference_batch_job"
        self.ml_supplier_po_item_inference_module = "ml_supplier_po_item_inference_module"
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")

        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')

        self.env = sbnutils._get_env()
        self.model_name = get_model_name(env=self.env)
        self.model_version = get_model_version(env=self.env)
        self.model_stage = get_model_stage(env = self.env)
        self.feature_concat_name = get_feature_concat_name(env = self.env)
        self.column_list = get_feature_concat_cols(env = self.env)
        self.na_value_list = get_na_value_list(env = self.env)
        self.replacement_value = get_replacement_value(env = self.env)

    # Case 1: Insert new data into bna_report table
    def before_ml_supplier_po_item_inference_1_first_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()
        timestamp_before_earliest = timestamp_earliest - datetime.timedelta(microseconds=1)
        timestamp_current = datetime.datetime.now()
        timestamp_after_current = timestamp_current + datetime.timedelta(minutes=40)

        df_model_info = spark.createDataFrame([
            ('MID-001', f'{self.model_name}', int(f'{self.model_version}'), f'{self.model_stage}', timestamp_earliest, timestamp_earliest)
        ], schema = self.ML_MODEL_INFO_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.GENERAL.value, MLTable.GENERAL_ML_MODEL_INFO.value).alias("t")
                .merge(df_model_info.alias("s"), "t.UUID = s.UUID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

        df_data = spark.createDataFrame([
            ('ID-10001', 10001, 'HP PAVILION 14 11TH GEN INTEL CORE I7 16GB',0, 100, 200, *(None for i in range(14)), timestamp_before_earliest, timestamp_before_earliest),
            ('ID-10002', 10002, 'BULLNOSE SHELVES 4 PK', 0, 100, 200, *(None for i in range(14)), timestamp_earliest, timestamp_earliest),
            ('ID-10003', 10003, 'MAIL FELINE 17 MM CENT FIX IND OR GR SI', 0, 100, 200, *(None for i in range(14)), timestamp_current, timestamp_current),
            ('ID-10004', 10004, 'ZZZ BULLNOSE SHELVES 3 PK', 0, 100, 200, *(None for i in range(14)), timestamp_after_current, timestamp_after_current),
            ('ID-10005', 10005, None, 0, 100, 200, *(None for i in range(14)), timestamp_earliest, timestamp_earliest),
        ], schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(df_data.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_inference_1_first_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_inference_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_inference_module, "tasklist": self.ml_supplier_po_item_inference_module}
        )

    def assertion_ml_supplier_po_item_inference_1_first_round(self):
        ml_po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        po_item_df = spark.sql(
                f"""SELECT * FROM {ml_po_item_table_full_name} t 
                WHERE SBN_PREDICTION_LASTUPDATED_AT IS NOT NULL
                ORDER BY t.ID ASC"""
            )
        # fill na values
        po_item_df = replace_columns_values(po_item_df, self.column_list, self.na_value_list, self.replacement_value)
        # add concat feature col
        po_item_df = add_concat_feature_column_to_df(po_item_df, self.env)

        bna_report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_BNA_UPSPSC_REPORT.value)
        bna_report_df = spark.sql(
            f"""SELECT * FROM {bna_report_table_full_name} ORDER BY {self.feature_concat_name}, {DELTA_UPDATED_FIELD}"""
        )
    
        joined_df = po_item_df.join(bna_report_df, on = [f'{self.feature_concat_name}', 'MODEL_UUID'], how='inner')
        joined_list = joined_df.collect()
        assert len(joined_list) == 2

        po_item_table_list = po_item_df.collect()
        assert len(po_item_table_list) == 2
        assert po_item_table_list[0]["ID"] == "ID-10002"
        assert po_item_table_list[0]["SBN_PREDICATED_UNSPSC_SEGMENT"] != None
        assert po_item_table_list[0]["SBN_PREDICTION_CONFIDENCE_SEGMENT"] >= 0
        assert po_item_table_list[0]["SBN_PREDICTION_LASTUPDATED_AT"] > po_item_table_list[0][DELTA_CREATED_FIELD]
        assert po_item_table_list[0]["MODEL_UUID"] == "MID-001"
        assert po_item_table_list[0][DELTA_UPDATED_FIELD] > po_item_table_list[0][DELTA_CREATED_FIELD]

        assert po_item_table_list[1]["ID"] == "ID-10003"
        assert po_item_table_list[1]["SBN_PREDICATED_UNSPSC_SEGMENT"] != None
        assert po_item_table_list[1]["SBN_PREDICTION_CONFIDENCE_SEGMENT"] >= 0
        assert po_item_table_list[1]["SBN_PREDICTION_LASTUPDATED_AT"] > po_item_table_list[1][DELTA_CREATED_FIELD]
        assert po_item_table_list[1]["MODEL_UUID"] == "MID-001"
        assert po_item_table_list[1][DELTA_UPDATED_FIELD] > po_item_table_list[1][DELTA_CREATED_FIELD]

        # Assert bna report table
        bna_report_table_list = bna_report_df.collect()
        assert len(bna_report_table_list) == 2
        assert bna_report_table_list[0][self.feature_concat_name] != "BULLNOSE SHELVES 4 PK"
        assert bna_report_table_list[0]["UNSPSC_SEGMENT"] != None
        assert bna_report_table_list[0]["CONFIDENCE_SEGMENT"] != None
        assert bna_report_table_list[0]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[0]["REPORT_LASTUPDATED_AT"] == bna_report_table_list[0][DELTA_UPDATED_FIELD]
        assert bna_report_table_list[0][DELTA_UPDATED_FIELD] == bna_report_table_list[0][DELTA_CREATED_FIELD]

        assert bna_report_table_list[1][self.feature_concat_name] != "MAIL FELINE 17 MM CENT FIX IND OR GR SI"
        assert bna_report_table_list[1]["UNSPSC_SEGMENT"] != None
        assert bna_report_table_list[1]["CONFIDENCE_SEGMENT"] != None
        assert bna_report_table_list[1]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[1]["REPORT_LASTUPDATED_AT"] == bna_report_table_list[0][DELTA_UPDATED_FIELD]
        assert bna_report_table_list[1][DELTA_UPDATED_FIELD] == bna_report_table_list[0][DELTA_CREATED_FIELD]

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_inference_module)

    # Case 2: Insert description(already in bna) into ml_supplier_po_item table, model_uuid = bna_report.model_uuid
    def before_ml_supplier_po_item_inference_2_second_round(self):
        timestamp_current = datetime.datetime.now()
        timestamp_before_current = timestamp_current - datetime.timedelta(minutes=20)
        # insert ml_supplier_po_item data
        df_data = spark.createDataFrame([
            ('ID-10004', 10004, 'ZZZ BULLNOSE SHELVES 3 PK', 0, 100, 200, *(None for _ in range(14)), timestamp_before_current, timestamp_before_current),
            ('ID-20001', 20001, 'BULLNOSE SHELVES 4 PK', 0, 100, 200, *(None for _ in range(14)), timestamp_current,timestamp_current), # won't be inference
            ('ID-20002', 20002, 'Z TEST CASE2 DESCRIPTION', 0, 100, 200, *(None for _ in range(14)), timestamp_current,timestamp_current) # will be inference
        ], schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
                .merge(df_data.alias("s"), "t.ID = s.ID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_inference_2_second_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_inference_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_inference_module, "tasklist": self.ml_supplier_po_item_inference_module}
        )


    def assertion_ml_supplier_po_item_inference_2_second_round(self):
        ml_po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        po_item_df = spark.sql(
                f"""SELECT * FROM {ml_po_item_table_full_name} t 
                WHERE SBN_PREDICTION_LASTUPDATED_AT IS NOT NULL
                ORDER BY t.ID ASC"""
            )
        # fill na values
        po_item_df = replace_columns_values(po_item_df, self.column_list, self.na_value_list, self.replacement_value)
        # add concat feature col
        po_item_df = add_concat_feature_column_to_df(po_item_df, self.env)

        bna_report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_BNA_UPSPSC_REPORT.value)
        bna_report_df = spark.sql(
            f"""SELECT * FROM {bna_report_table_full_name} ORDER BY {self.feature_concat_name}, {DELTA_UPDATED_FIELD}"""
        )
    
        joined_df = (
            po_item_df.join(bna_report_df, on = [f'{self.feature_concat_name}', 'MODEL_UUID'], how='inner')
                      .orderBy('ID')
        )
        joined_list = joined_df.collect()
        assert len(joined_list) == 4
        assert joined_list[0][self.feature_concat_name] == joined_list[2][self.feature_concat_name]

        po_item_table_list = po_item_df.collect()
        assert len(po_item_table_list) == 4
        assert po_item_table_list[2]["ID"] == "ID-20001" # not inference
        assert po_item_table_list[2]["SBN_PREDICATED_UNSPSC_SEGMENT"] != None
        assert po_item_table_list[2]["SBN_PREDICTION_CONFIDENCE_SEGMENT"] >= 0
        assert po_item_table_list[2]["MODEL_UUID"] == "MID-001"
        assert po_item_table_list[2]["SBN_PREDICTION_LASTUPDATED_AT"] < po_item_table_list[2][DELTA_UPDATED_FIELD]
        assert po_item_table_list[2][DELTA_UPDATED_FIELD] > po_item_table_list[2][DELTA_CREATED_FIELD]

        assert po_item_table_list[3]["ID"] == "ID-20002" # inferenced
        assert po_item_table_list[3]["SBN_PREDICATED_UNSPSC_SEGMENT"] != None
        assert po_item_table_list[3]["SBN_PREDICTION_CONFIDENCE_SEGMENT"] >= 0
        assert po_item_table_list[3]["MODEL_UUID"] == "MID-001"
        assert po_item_table_list[3]["SBN_PREDICTION_LASTUPDATED_AT"] == po_item_table_list[3][DELTA_UPDATED_FIELD]
        assert po_item_table_list[3][DELTA_UPDATED_FIELD] > po_item_table_list[3][DELTA_CREATED_FIELD]

        # Assert bna report table
        bna_report_table_list = bna_report_df.collect()
        assert len(bna_report_table_list) == 3
        assert bna_report_table_list[0][self.feature_concat_name] != "BULLNOSE SHELVES 4 PK" # won't update
        assert bna_report_table_list[0]["UNSPSC_SEGMENT"] != None
        assert bna_report_table_list[0]["CONFIDENCE_SEGMENT"] != None
        assert bna_report_table_list[0]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[0]["REPORT_LASTUPDATED_AT"] == bna_report_table_list[0][DELTA_UPDATED_FIELD]
        assert bna_report_table_list[0][DELTA_UPDATED_FIELD] == bna_report_table_list[0][DELTA_CREATED_FIELD]

        assert bna_report_table_list[2][self.feature_concat_name] != "Z TEST CASE2 DESCRIPTION" # new row
        assert bna_report_table_list[2]["UNSPSC_SEGMENT"] != None
        assert bna_report_table_list[2]["CONFIDENCE_SEGMENT"] != None
        assert bna_report_table_list[2]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[2]["REPORT_LASTUPDATED_AT"] == bna_report_table_list[2][DELTA_UPDATED_FIELD]
        assert bna_report_table_list[2][DELTA_UPDATED_FIELD] == bna_report_table_list[2][DELTA_CREATED_FIELD]

        assert bna_report_table_list[0][DELTA_UPDATED_FIELD] < bna_report_table_list[2][DELTA_UPDATED_FIELD]

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_inference_module)

    # 3. Case 3: reprocessing, model_uuid != bna_report.model_uuid
    def before_ml_supplier_po_item_inference_3_third_round(self):
        timestamp_earliest = batch_utils.get_timestamp_earliest()

        # update the model info uuid, imagine we are using a new model, and the bna_report data are out of date
        df_model_info = spark.createDataFrame([
            ('MID-001', 'case_1_model', 1, 'Staging', timestamp_earliest, timestamp_earliest),
            ('MID-302', f'{self.model_name}', int(f'{self.model_version}'), f'{self.model_stage}', timestamp_earliest, timestamp_earliest)
        ], schema=self.ML_MODEL_INFO_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.GENERAL.value, MLTable.GENERAL_ML_MODEL_INFO.value).alias("t")
                .merge(df_model_info.alias("s"), "t.UUID = s.UUID")
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
        )

    def run_ml_supplier_po_item_inference_3_third_round(self):
        # Update reset flag to get ready to reprocess data
        batch_utils.update_offset_timestamp_reset_flag(self.ml_supplier_po_item_inference_module, 1)

        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_inference_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_inference_module, "tasklist": self.ml_supplier_po_item_inference_module}
        )


    def assertion_ml_supplier_po_item_inference_3_third_round(self):
        ml_po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        po_item_df = spark.sql(
                f"""SELECT * FROM {ml_po_item_table_full_name} t 
                WHERE SBN_PREDICTION_LASTUPDATED_AT IS NOT NULL
                ORDER BY t.ID ASC"""
            )
        # fill na values
        po_item_df = replace_columns_values(po_item_df, self.column_list, self.na_value_list, self.replacement_value)
        # add concat feature col
        po_item_df = add_concat_feature_column_to_df(po_item_df, self.env)

        bna_report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_BNA_UPSPSC_REPORT.value)
        bna_report_df = spark.sql(
            f"""SELECT * FROM {bna_report_table_full_name} ORDER BY {self.feature_concat_name}, {DELTA_UPDATED_FIELD}"""
        )
    
        joined_df = (
            po_item_df.join(bna_report_df, on = [f'{self.feature_concat_name}', 'MODEL_UUID'], how='inner')
                      .orderBy('ID')
        )
        joined_list = joined_df.collect()
        assert len(joined_list) == 5

        po_item_table_list = po_item_df.collect()
        assert len(po_item_table_list) == 5
        assert po_item_table_list[0]["ID"] == "ID-10002"
        assert po_item_table_list[0]["MODEL_UUID"] == "MID-302"  # use new model inference
        assert po_item_table_list[0][DELTA_UPDATED_FIELD] == po_item_table_list[0]["SBN_PREDICTION_LASTUPDATED_AT"]
        assert po_item_table_list[0][DELTA_UPDATED_FIELD] > po_item_table_list[0][DELTA_CREATED_FIELD]

        assert po_item_table_list[1]["ID"] == "ID-10003"
        assert po_item_table_list[1]["MODEL_UUID"] == "MID-302"
        assert po_item_table_list[1][DELTA_UPDATED_FIELD] > po_item_table_list[1][DELTA_CREATED_FIELD]

        assert po_item_table_list[2]["ID"] == "ID-10004"  # new data
        assert po_item_table_list[2]["MODEL_UUID"] == "MID-302"
        assert po_item_table_list[2][DELTA_UPDATED_FIELD] == po_item_table_list[2]["SBN_PREDICTION_LASTUPDATED_AT"]
        assert po_item_table_list[2][DELTA_UPDATED_FIELD] > po_item_table_list[2][DELTA_CREATED_FIELD]

        assert po_item_table_list[3]["ID"] == "ID-20001"
        assert po_item_table_list[3]["MODEL_UUID"] == "MID-302"
        assert po_item_table_list[3][DELTA_UPDATED_FIELD] > po_item_table_list[3][DELTA_CREATED_FIELD]

        assert po_item_table_list[4]["ID"] == "ID-20002"
        assert po_item_table_list[4]["SBN_PREDICATED_UNSPSC_SEGMENT"] != None
        assert po_item_table_list[4]["SBN_PREDICTION_CONFIDENCE_SEGMENT"] >= 0
        assert po_item_table_list[4]["SBN_PREDICTION_LASTUPDATED_AT"] == po_item_table_list[4][DELTA_UPDATED_FIELD]
        assert po_item_table_list[4]["MODEL_UUID"] == "MID-302" # use new model inference
        assert po_item_table_list[4][DELTA_UPDATED_FIELD] > po_item_table_list[4][DELTA_CREATED_FIELD]

        # Assert bna report table
        bna_report_table_list = bna_report_df.collect()
        assert len(bna_report_table_list) == 7
        assert bna_report_table_list[0][self.feature_concat_name] != "BULLNOSE SHELVES 4 PK"
        assert bna_report_table_list[0]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[1][self.feature_concat_name] == bna_report_table_list[0][self.feature_concat_name]
        assert bna_report_table_list[1]["MODEL_UUID"] == "MID-302"
        assert bna_report_table_list[1][DELTA_UPDATED_FIELD] > bna_report_table_list[0][DELTA_UPDATED_FIELD]

        assert bna_report_table_list[2][self.feature_concat_name] != "MAIL FELINE 17 MM CENT FIX IND OR GR SI"
        assert bna_report_table_list[2]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[3][self.feature_concat_name] == bna_report_table_list[2][self.feature_concat_name]
        assert bna_report_table_list[3]["MODEL_UUID"] == "MID-302"
        assert bna_report_table_list[3][DELTA_UPDATED_FIELD] > bna_report_table_list[2][DELTA_UPDATED_FIELD]

        assert bna_report_table_list[4][self.feature_concat_name] != "Z TEST CASE2 DESCRIPTION"
        assert bna_report_table_list[4]["MODEL_UUID"] == "MID-001"
        assert bna_report_table_list[5][self.feature_concat_name] == bna_report_table_list[4][self.feature_concat_name]
        assert bna_report_table_list[5]["MODEL_UUID"] == "MID-302"
        assert bna_report_table_list[5][DELTA_UPDATED_FIELD] > bna_report_table_list[4][DELTA_UPDATED_FIELD]

        assert bna_report_table_list[6][self.feature_concat_name] != "ZZZ BULLNOSE SHELVES 3 PK"
        assert bna_report_table_list[6]["UNSPSC_SEGMENT"] != None
        assert bna_report_table_list[6]["CONFIDENCE_SEGMENT"] != None
        assert bna_report_table_list[6]["MODEL_UUID"] == "MID-302"
        assert bna_report_table_list[6]["REPORT_LASTUPDATED_AT"] == bna_report_table_list[6][DELTA_UPDATED_FIELD]
        assert bna_report_table_list[6][DELTA_UPDATED_FIELD] == bna_report_table_list[6][DELTA_CREATED_FIELD]

        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_inference_module)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = MlSupplierPoItemInferenceBatchFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)

# COMMAND ----------