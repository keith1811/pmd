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
import sys
import os
import sqls
from integration_tests.quality_utils import SbnNutterFixture

from utils import *
from utils.constants import Zone
from decimal import *

from modules.utils.constants import MLTable

# COMMAND ----------

class PredictionUpdateFromGdsFixture(SbnNutterFixture):
    ML_GDS_UNSPSC_REPORT_SCHEMA = """
    DESCRIPTION STRING, 
    PROCESSED_DESCRIPTION STRING, 
    UNSPSC_SEGMENT STRING,
    UNSPSC_FAMILY STRING,
    UNSPSC_CLASS STRING,
    UNSPSC_COMMODITY STRING,
    CONFIDENCE_SEGMENT DECIMAL(6,5),
    CONFIDENCE_FAMILY DECIMAL(6,5),
    CONFIDENCE_CLASS DECIMAL(6,5),
    CONFIDENCE_COMMODITY DECIMAL(6,5),
    REPORT_LASTUPDATED_AT TIMESTAMP"""

    ML_SUPPLIER_PO_ITEM_SCHEMA = """
    ID STRING,
    PO_ITEM_ID DECIMAL(28,0),
    DESCRIPTION STRING, 
    PROCESSED_DESCRIPTION STRING, 
    EXTERNAL_PREDICATED_UNSPSC_SEGMENT STRING,
    EXTERNAL_PREDICATED_UNSPSC_FAMILY STRING,
    EXTERNAL_PREDICATED_UNSPSC_CLASS STRING,
    EXTERNAL_PREDICATED_UNSPSC_COMMODITY STRING,
    EXTERNAL_PREDICATION_CONFIDENCE_SEGMENT DECIMAL(6,5),
    EXTERNAL_PREDICATION_CONFIDENCE_FAMILY DECIMAL(6,5),
    EXTERNAL_PREDICATION_CONFIDENCE_CLASS DECIMAL(6,5),
    EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY DECIMAL(6,5),
    EXTERNAL_PREDICATION_LASTUPDATED_AT TIMESTAMP,
    _DELTA_CREATED_ON TIMESTAMP,
    _DELTA_UPDATED_ON TIMESTAMP"""
    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "it_ml_supplier_po_item_gds_update_job"
        self.ml_supplier_po_item_gds_update_module = "ml_supplier_po_item_gds_update_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')
        self.timestamp_current = datetime.datetime.now()
        self.timestamp_before_current = self.timestamp_current - datetime.timedelta(minutes=60)
        self.timestamp_after_current = self.timestamp_current + datetime.timedelta(minutes=60)


    # 1. first round: first batch
    def before_ml_supplier_po_item_gds_update_1_first_round(self):

        # insert test data into ml_supplier_po_item
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ('24324', Decimal('5'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)), self.timestamp_before_current, self.timestamp_before_current),
            ('32114', Decimal('9'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)), self.timestamp_before_current, self.timestamp_before_current),
            ('42424', Decimal('6'), 'KASFNKA DJAFKA DKJNAK', 'KASFNKA DJAFKA DKJNAK', *(None for i in range(9)), self.timestamp_before_current, self.timestamp_before_current),
            ('55213', Decimal('7'), 'ABAK DJAN DDJ DJAS', 'ABAK DJAN DDJ DJAS', *(None for i in range(9)), self.timestamp_before_current, self.timestamp_before_current),
            ('21484', Decimal('10'), 'SHFKJAFH.SJJ DKAHD', 'SHFKJAFH.SJJ DKAHD', *(None for i in range(9)), self.timestamp_before_current, self.timestamp_before_current),
            ('14242', Decimal('8'), 'JSAFN FJSAK ASJDN DKA', 'JSAFN FJSAK ASJDN DKA', *(None for i in range(9)), self.timestamp_before_current, self.timestamp_before_current)],
            schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        po_item_source_location = sbnutils.get_table_storage_location(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        ml_supplier_po_item_test_df.write.format("delta").mode("overwrite").save(po_item_source_location)

        # insert test data into ml_supplier_po_item_gds
        ml_gds_unspsc_report_test_df = spark.createDataFrame([
            ('DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', '10', '1010', '101015', '1010101520', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), datetime.datetime.now()),
            ('KASFNKA DJAFKA DKJNAK', 'KASFNKA DJAFKA DKJNAK', '10', '1011', '101116', '1010101522', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), datetime.datetime.now()),
            ('ABAK DJAN DDJ DJAS', 'ABAK DJAN DDJ DJAS', '10', '1012', '101017', '1010101524', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), datetime.datetime.now()),
            ('JSAFN FJSAK ASJDN DKA', 'JSAFN FJSAK ASJDN DKA', '10', '1013', '101018', '1010101526', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), datetime.datetime.now()),
            ('FJDKS SKNFDS DJI SSN', 'FJDKS SKNFDS DJI SSN', '10', '1012', '101017', '1010101524', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), datetime.datetime.now())],
            schema=self.ML_GDS_UNSPSC_REPORT_SCHEMA)
        gds_source_location = sbnutils.get_table_storage_location(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value)
        ml_gds_unspsc_report_test_df.write.format("delta").mode("overwrite").save(gds_source_location)

        
    def run_ml_supplier_po_item_gds_update_1_first_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_gds_update_module, "tasklist": self.ml_supplier_po_item_gds_update_module},
        )

    def assertion_ml_supplier_po_item_gds_update_1_first_round(self):
        # Assert target file
        ml_po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        ml_po_item_target_table_list = spark.sql(
            f"SELECT * FROM {ml_po_item_table_full_name} t ORDER BY t.PO_ITEM_ID"
        ).collect()
        assert len(ml_po_item_target_table_list) == 6
        assert ml_po_item_target_table_list[0]["DESCRIPTION"] == 'DJSIFN SJFSJKS SKAFNKSFN'
        assert ml_po_item_target_table_list[0]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '10'

        unspsc_report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value)
        unspsc_report_target_table_list = spark.sql(
            f"SELECT * FROM {unspsc_report_table_full_name} t ORDER BY t.DESCRIPTION"
        ).collect()
        assert len(unspsc_report_target_table_list) == 5
        assert unspsc_report_target_table_list[0]["DESCRIPTION"] == 'ABAK DJAN DDJ DJAS'
        assert unspsc_report_target_table_list[0]["UNSPSC_SEGMENT"] == '10'


    # 2. second round: first batch
    def before_ml_supplier_po_item_gds_update_2_second_round(self):
        # insert test data into ml_supplier_po_item
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ('24324', Decimal('5'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', '10', '1010', '101015', '1010101520', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), self.timestamp_current, self.timestamp_before_current, datetime.datetime.now()),
            ('32114', Decimal('6'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)), self.timestamp_before_current, datetime.datetime.now()),
            ('24245', Decimal('7'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)), self.timestamp_before_current, datetime.datetime.now()),
            ('47474', Decimal('8'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)), self.timestamp_before_current, datetime.datetime.now()),
            ('45346', Decimal('9'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)),  self.timestamp_before_current, datetime.datetime.now()),
            ('89644', Decimal('10'), 'DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', *(None for i in range(9)), self.timestamp_before_current, datetime.datetime.now()),
            ('42424', Decimal('11'), 'KASFNKA DJAFKA DKJNAK', 'KASFNKA DJAFKA DKJNAK', '10', '1010', '101015', '1010101520', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), self.timestamp_current, self.timestamp_before_current, self.timestamp_before_current),
            ('14242', Decimal('12'), 'JSAFN FJSAK ASJDN DKA', 'JSAFN FJSAK ASJDN DKA', '10', '1010', '101015', '1010101520', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), self.timestamp_current, self.timestamp_before_current, self.timestamp_before_current)],
            schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        po_item_source_location = sbnutils.get_table_storage_location(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        ml_supplier_po_item_test_df.write.format("delta").mode("overwrite").save(po_item_source_location)

        # insert test data into ml_supplier_po_item_gds
        ml_gds_unspsc_report_test_df = spark.createDataFrame([
            ('DJSIFN SJFSJKS SKAFNKSFN', 'DJSIFN SJFSJKS SKAFNKSFN', '20', '1010', '101015', '1010101520', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), self.timestamp_current),
            ('KASFNKA DJAFKA DKJNAK', 'KASFNKA DJAFKA DKJNAK', '10', '1011', '101116', '1010101522', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), self.timestamp_current),
            ('JSAFN FJSAK ASJDN DKA', 'JSAFN FJSAK ASJDN DKA', '10', '1013', '101018', '1010101526', Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), Decimal('1.12343'), self.timestamp_current)],
            schema=self.ML_GDS_UNSPSC_REPORT_SCHEMA)
        gds_source_location = sbnutils.get_table_storage_location(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value)
        ml_gds_unspsc_report_test_df.write.format("delta").mode("overwrite").save(gds_source_location)

    
    def run_ml_supplier_po_item_gds_update_2_second_round(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.ml_supplier_po_item_gds_update_module, "tasklist": self.ml_supplier_po_item_gds_update_module},
        )

    def assertion_ml_supplier_po_item_gds_update_2_second_round(self):
        # Assert target file
        ml_po_item_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
        ml_po_item_target_table_list = spark.sql(
            f"SELECT * FROM {ml_po_item_table_full_name} t ORDER BY t.PO_ITEM_ID"
        ).collect()
        assert len(ml_po_item_target_table_list) == 8
        assert ml_po_item_target_table_list[0]["ID"] == '24324'
        assert ml_po_item_target_table_list[0]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '10'

        assert ml_po_item_target_table_list[1]["ID"] == '32114'
        assert ml_po_item_target_table_list[1]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '20'

        assert ml_po_item_target_table_list[4]["ID"] == '45346'
        assert ml_po_item_target_table_list[4]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '20'

        assert ml_po_item_target_table_list[5]["ID"] == '89644'
        assert ml_po_item_target_table_list[5]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '20'

        unspsc_report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value)
        unspsc_report_target_table_list = spark.sql(
            f"SELECT * FROM {unspsc_report_table_full_name} t ORDER BY t.DESCRIPTION"
        ).collect()
        assert len(unspsc_report_target_table_list) == 3
        assert unspsc_report_target_table_list[0]["DESCRIPTION"] == 'DJSIFN SJFSJKS SKAFNKSFN'
        assert unspsc_report_target_table_list[0]["UNSPSC_SEGMENT"] == '20'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run test cases

# COMMAND ----------

result = PredictionUpdateFromGdsFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)

# COMMAND ----------


