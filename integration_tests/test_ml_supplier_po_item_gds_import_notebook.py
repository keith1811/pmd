# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - Prediction Read From GDS job

# COMMAND ----------

# MAGIC %md
# MAGIC #### Setup

# COMMAND ----------

!cp ../requirements.txt ~/.
%pip install -r ~/requirements.txt
%load_ext autoreload
%autoreload 2

# COMMAND ----------

import datetime
import os
from decimal import *

from integration_tests.quality_utils import SbnNutterFixture
from utils import *
from utils.constants import Zone

import sqls
from modules.utils.constants import MLTable


# COMMAND ----------

class PredictionReadFromGdsFixture(SbnNutterFixture):
    ML_SUPPLIER_PO_ITEM_GDS_SCHEMA = """
    DESCRIPTION STRING, 
    PROCESSED_DESCRIPTION STRING, 
    PREDICATED_UNSPSC_SEGMENT STRING,
    PREDICATED_UNSPSC_FAMILY STRING,
    PREDICATED_UNSPSC_CLASS STRING,
    PREDICATED_UNSPSC_COMMODITY STRING,
    PREDICATION_LASTUPDATED_AT TIMESTAMP"""

    ML_SUPPLIER_PO_ITEM_SCHEMA = """
    ID STRING,
    PO_ITEM_ID DECIMAL(28,0),
    DESCRIPTION STRING, 
    PROCESSED_DESCRIPTION STRING, 
    EXTERNAL_PREDICATED_UNSPSC_SEGMENT STRING,
    EXTERNAL_PREDICATED_UNSPSC_FAMILY STRING,
    EXTERNAL_PREDICATED_UNSPSC_CLASS STRING,
    EXTERNAL_PREDICATED_UNSPSC_COMMODITY STRING,
    EXTERNAL_PREDICATION_LASTUPDATED_AT TIMESTAMP"""
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_gds_import_job"
        self.ml_supplier_po_item_gds_import_module = "ml_supplier_po_item_gds_import_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')


    # 1. first round: first batch
    def before_ml_supplier_po_item_gds_import_1_first_round(self):

        # insert test data into ml_supplier_po_item_gds
        ml_supplier_po_item_gds_test_df = spark.createDataFrame([
            ('DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', '10', '1010', '101015', '1010101520', datetime.datetime.now()),
            ('KASFNKA DJAFKA DKJNAK', 'KASFNKA DJAFKA DKJNAK', '10', '1011', '101116', '1010101522', datetime.datetime.now()),
            ('ABAK DJAN DDJ DJAS', 'ABAK DJAN DDJ DJAS', '10', '1012', '101017', '1010101524', datetime.datetime.now()),
            ('JSAFN FJSAK ASJDN DKA', 'JSAFN FJSAK ASJDN DKA', '10', '1013', '101018', '1010101526', datetime.datetime.now()),
            ('ABAK DJAN DDJ DJAS', 'ABAK DJAN DDJ DJAS', '10', '1012', '101017', '1010101524', datetime.datetime.now())],
            schema=self.ML_SUPPLIER_PO_ITEM_GDS_SCHEMA)
        gds_source_location = sbnutils.get_table_storage_location(Zone.STAGING.value, MLTable.STAGING_ML_UNSPSC_REPORT_GDS.value)
        ml_supplier_po_item_gds_test_df.write.format("csv").mode("overwrite").option("header", "true").save(gds_source_location)

        # insert test data into ml_supplier_po_item
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ('24324', Decimal('5'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(5))),
            ('32114', Decimal('9'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(5))),
            ('42424', Decimal('6'), 'KASFNKA DJAFKA DKJNAK', 'KASFNKA DJAFKA DKJNAK', *(None for i in range(5))),
            ('55213', Decimal('7'), 'ABAK DJAN DDJ DJAS', 'ABAK DJAN DDJ DJAS', *(None for i in range(5))),
            ('14242', Decimal('8'), 'JSAFN FJSAK ASJDN DKA', 'JSAFN FJSAK ASJDN DKA', *(None for i in range(5)))],
            schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        source_location = sbnutils.get_table_storage_location(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        ml_supplier_po_item_test_df.write.format("delta").mode("overwrite").save(source_location)

    def run_ml_supplier_po_item_gds_import_1_first_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_gds_import_module, "tasklist": self.ml_supplier_po_item_gds_import_module, "need_archive": "true"},
        )

    def assertion_ml_supplier_po_item_gds_import_1_first_round(self):
        # Assert target file
        ml_po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        ml_po_item_target_table_list = spark.sql(
            f"SELECT * FROM {ml_po_item_table_full_name} t ORDER BY t.PO_ITEM_ID"
        ).collect()
        assert len(ml_po_item_target_table_list) == 5
        assert ml_po_item_target_table_list[0]["DESCRIPTION"] == 'DJSIFN SJFSJKS SKAFNKSFN'
        assert ml_po_item_target_table_list[1]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '10'

        unspsc_report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value)
        unspsc_report_target_table_list = spark.sql(
            f"SELECT * FROM {unspsc_report_table_full_name} t ORDER BY t.DESCRIPTION"
        ).collect()
        assert len(unspsc_report_target_table_list) == 4
        assert unspsc_report_target_table_list[0]["DESCRIPTION"] == 'ABAK DJAN DDJ DJAS'
        assert unspsc_report_target_table_list[1]["UNSPSC_SEGMENT"] == '10'


        # Assert job config table
        self.assert_job_config_table(self.ml_supplier_po_item_gds_import_module)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = PredictionReadFromGdsFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
