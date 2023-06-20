from pyspark.sql.functions import current_timestamp
from datetime import datetime

from utils import sbnutils
from utils import batch_utils
from utils import constants
from utils.constants import Zone
from modules.utils.constants import MLTable

def main():
    # Job Start
    sbnutils.log_info(f"Export ml_supplier_po_item batch job Start")

    source_zone = Zone.ENRICHMENT.value
    source_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    target_zone = Zone.SAP_EXPORT.value
    target_file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    report_zone = Zone.ENRICHMENT.value
    report_table_name = MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value

    source_table_location = sbnutils.get_table_storage_location(source_zone, source_table_name)
    report_table_location = sbnutils.get_table_storage_location(report_zone, report_table_name)
    report_table = sbnutils.get_delta_table(report_zone, report_table_name)
    storage_location_dir = f'{sbnutils._get_location_base(target_zone)}gds'
    sbnutils.log_info(f"target_storage_location_dir: {storage_location_dir}")

    # Read the source table data as a batch
    source_df = _read_data(source_table_location)
    # Filter out exported data
    filterd_source_df = _filter_out_exported_data(source_df, report_table_location)
    # move to archive if exists
    _mv_to_archive(storage_location_dir)
    # write csv files to gds
    _write_csv(filterd_source_df, storage_location_dir, target_file_name)
    # merge data to ml_gds_unspsc_report
    _write_table(filterd_source_df, report_table)

    # Job End
    sbnutils.log_info(f"Export ml_supplier_po_item batch job End")

def _read_data(source_table_location):
    spark = sbnutils.get_spark()
    timestamp_range = batch_utils.get_batch_timestamp_range()

    # Read the table data as a batch
    source_df = (
        spark.read.format("delta")
        .load(source_table_location)
        .where(
            f"""
            {constants.DELTA_UPDATED_FIELD} >= '{timestamp_range[0]}' AND 
            {constants.DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}' AND
            EXTERNAL_PREDICATED_UNSPSC_SEGMENT is null AND
            EXTERNAL_PREDICATED_UNSPSC_FAMILY is null AND
            EXTERNAL_PREDICATED_UNSPSC_CLASS is null AND
            EXTERNAL_PREDICATED_UNSPSC_COMMODITY is null AND
            DASHBOARD_STATUS != 'PO_OBSOLETED' AND
            AN_DATA_QUALITY_LEVEL = 'Good'
            """
        )
        .select('PO_ITEM_ID', 'DESCRIPTION', 'PROCESSED_DESCRIPTION', 'MANUFACTURER_PART', 'MANUFACTURER_NAME', 'AN_UNSPSC_COMMODITY')
        .withColumn(constants.DELTA_UPDATED_FIELD, current_timestamp())
        .dropDuplicates(['PROCESSED_DESCRIPTION'])
    )

    return source_df

def _filter_out_exported_data(source_df, report_table_location):
    dbutils = sbnutils.get_dbutils()
    batch_size = dbutils.widgets.get("batch_size")
    sbnutils.log_info(f"Batch size: {batch_size}")

    exported_df = (
        sbnutils.get_spark().read.format("delta")
        .load(report_table_location)
        .select("PROCESSED_DESCRIPTION")
    )
    filterd_df = (
        source_df.join(exported_df, source_df.PROCESSED_DESCRIPTION == exported_df.PROCESSED_DESCRIPTION, "left_anti")
    )

    if batch_size:
        filterd_df = filterd_df.limit(int(batch_size))

    return filterd_df

def _write_csv(source_df, storage_location_dir, file_name):
    sbnutils.get_dbutils().fs.rm(storage_location_dir, True)
    sbnutils.get_dbutils().fs.mkdirs(storage_location_dir)
    source_df.write.mode("overwrite").option("header", "true").csv(f'{storage_location_dir}/{file_name}_{datetime.now().strftime("%Y_%m_%d")}')

def _write_table(source_df, target_table):
   feature_df = source_df.select('DESCRIPTION', 'PROCESSED_DESCRIPTION')
   # Write the data to the delta table
   sbnutils.write_data(feature_df, target_table, "PROCESSED_DESCRIPTION")

def _mv_to_archive(storage_location_dir):
    dbutils = sbnutils.get_dbutils()
    need_archive = dbutils.widgets.get("need_archive")
    sbnutils.log_info(f"Need archive: {need_archive}")

    if need_archive and need_archive.lower() == "true":
        sbnutils.get_dbutils().fs.mkdirs(storage_location_dir)
        filenames = list(map(lambda fileInfo:fileInfo.name, sbnutils.get_dbutils().fs.ls(storage_location_dir)))
        # if exists exoprt data need to archive
        if len(filenames):
            archive_dir = f'{storage_location_dir}_archive'
            sbnutils.get_dbutils().fs.mkdirs(archive_dir)

            filename = filenames[0]
            slash_index = filename.find('/')
            if slash_index != -1:
                filename = filename[: slash_index]
            filenames_in_archive = list(map(lambda fileInfo: fileInfo.name if fileInfo.name.find("/") == -1 
                                            else fileInfo.name[:fileInfo.name.find("/")], sbnutils.get_dbutils().fs.ls(archive_dir)))
            # get a different archive_filename from the other files
            archive_filename = filename
            i = 0
            while f'{archive_filename}' in filenames_in_archive:
                i += 1
                archive_filename = f'{filename}_({i})'
            sbnutils.get_dbutils().fs.mv(f'{storage_location_dir}/{filename}', f'{archive_dir}/{archive_filename}', True)
