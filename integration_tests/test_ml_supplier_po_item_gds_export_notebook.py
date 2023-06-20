# Databricks notebook source
# MAGIC %md
# MAGIC ### Integration tests - Export ml_supplier_po_item job

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

from utils import sbnutils
from utils import batch_utils
from utils import database_utils
from utils.constants import Zone
from utils.constants import Table

from modules.utils.constants import MLTable

# COMMAND ----------

class ExportMLSupplierPOItemFixture(SbnNutterFixture):
    ML_SUPPLIER_PO_ITEM_SCHEMA = """ ID string, PO_ITEM_ID long, DASHBOARD_STATUS string, AN_DATA_QUALITY_LEVEL string, DESCRIPTION string, PROCESSED_DESCRIPTION string, MANUFACTURER_PART string, MANUFACTURER_NAME string, AN_UNSPSC_COMMODITY string, 
    EXTERNAL_PREDICATED_UNSPSC_SEGMENT string, _DELTA_CREATED_ON TIMESTAMP, _DELTA_UPDATED_ON TIMESTAMP """

    def __init__(self):
        """
        Caution: This case name will be used to name the temporary database in a sandboxed environment.
                 ONLY lowercase letters, numbers and underscores are allowed.
        Temporary database name pattern: IT_{commit_id*}_{case_name}_{commit_timestamp}
            *: If run this IT notebook directly, the commit_id will be set to 'notebook_run' by default.
        """
        case_name = "itcase_ml_supplier_po_item_gds_export_job"
        self.import_module = "ml_supplier_po_item_gds_export_module"
        SbnNutterFixture.__init__(self, case_name)
        database_utils.execute_ddl_sql(Zone.GENERAL.value, f'{os.path.dirname(sqls.__file__)}/general')
        database_utils.execute_ddl_sql(Zone.ENRICHMENT.value, f'{os.path.dirname(sqls.__file__)}/enrichment')
        self.timestamp_current = datetime.datetime.now()

    # 1. Test exort_no_extra_parameters
    def before_ml_supplier_po_item_export_1_archive_one(self):
        # Add the test data to the source table
        spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "True")
        ml_supplier_po_item_test_df = spark.createDataFrame([
            ("1", 1001, 'PO_NEW', 'Good', 'aa/', 'aa', 'mp', 'mn', '12345678', None, self.timestamp_current, self.timestamp_current),
            ("2", 1002, 'PO_SHIPPED', 'Poor','b//b@', 'bb', 'mp', 'mn', '12345678', None, self.timestamp_current, self.timestamp_current),
            ], schema=self.ML_SUPPLIER_PO_ITEM_SCHEMA)
        (
            sbnutils.get_delta_table(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value).alias("t")
            .merge(ml_supplier_po_item_test_df.alias("s"), "t.ID = s.ID")
            .whenNotMatchedInsertAll()
            .execute()
        )
        file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        df = spark.createDataFrame([
            ('aa/', 'aa')
        ], schema='DESCRIPTION string, PROCESSED_DESCRIPTION string')
        (
            df.coalesce(1).write.format("csv").mode('overwrite').option("header", "true")
            .save(f'{sbnutils._get_location_base(Zone.SAP_EXPORT.value)}gds/{file_name}_{datetime.datetime.now().strftime("%Y_%m_%d")}')
        )

    def run_ml_supplier_po_item_export_1_archive_one(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.import_module, "tasklist": self.import_module, "need_archive": "true"},
        )

    def assertion_ml_supplier_po_item_export_1_archive_one(self):
        # Assert target file
        target_zone = Zone.SAP_EXPORT.value
        target_file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        target_storage_location = f"{sbnutils._get_location_base(target_zone)}gds/{target_file_name}_*"
        target_table_list = spark.read.format("csv").options(header='true').load(target_storage_location).collect()
        assert len(target_table_list) == 1
        assert len(target_table_list[0]) == 7
        assert target_table_list[0]['PROCESSED_DESCRIPTION'] == 'aa'

        report_table_full_name = sbnutils.get_table_full_name(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value)
        report_table_list = spark.sql(f"SELECT * FROM {report_table_full_name}").collect()
        assert len(report_table_list) == 1
        assert report_table_list[0]["DESCRIPTION"] == 'aa/'
        assert report_table_list[0]["PROCESSED_DESCRIPTION"] == 'aa'
        self.assert_archive_file(target_zone, target_file_name, 1)

    # 2. Test exort_with_batch_size_and_archive
    def before_ml_supplier_po_item_export_2_archive_two(self):
        pass

    def run_ml_supplier_po_item_export_2_archive_two(self):
        # Run batch job
        dbutils.notebook.run(
            "../notebooks/run_batch_job_notebook",
            1200,
            {**self.base_arguments, "import_module": self.import_module, "need_archive": "true", "tasklist": self.import_module},
        )

    def assertion_ml_supplier_po_item_export_2_archive_two(self):
        # Assert target file
        target_zone = Zone.SAP_EXPORT.value
        target_file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        target_storage_location = f"{sbnutils._get_location_base(target_zone)}gds/{target_file_name}_*"
        target_table_list = spark.read.format("csv").options(header='true').load(target_storage_location).collect()
        assert len(target_table_list) == 0
        self.assert_archive_file(target_zone, target_file_name, 2)

    def assert_archive_file(self, zone, file_name, number_of_archive_files):
        archive_dir = f"sap_export_gds_archive/"
        assert archive_dir in list(map(lambda fileInfo:fileInfo.name, sbnutils.get_dbutils().fs.ls(sbnutils._get_integration_test_info()[1])))
        archive_storage_account = f"{sbnutils._get_integration_test_info()[1]}/{archive_dir}"
        filenames = list(map(lambda fileInfo:fileInfo.name, sbnutils.get_dbutils().fs.ls(archive_storage_account)))
        assert len(filenames) == number_of_archive_files
        assert filenames[0] == f'{file_name}_{datetime.datetime.now().strftime("%Y_%m_%d")}/'
        for i in range(1, number_of_archive_files):
            assert filenames[i] == f'{file_name}_{datetime.datetime.now().strftime("%Y_%m_%d")}_({i})/'


# COMMAND ----------

result = ExportMLSupplierPOItemFixture().execute_tests()
print(result.to_string())
# Comment out the next line (result.exit(dbutils)) to see the test result report from within the notebook
result.exit(dbutils)
