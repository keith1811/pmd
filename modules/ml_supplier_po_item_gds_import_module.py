from pyspark.sql.functions import current_timestamp, row_number, desc, col
from pyspark.sql.window import Window

from modules.utils.constants import MLTable
from utils import sbnutils
from utils.constants import DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from utils.constants import Zone

PO_ITEM_COLUMNS_EXTERNAL = [
    "DESCRIPTION",
    "PROCESSED_DESCRIPTION",
    "PREDICATED_UNSPSC_SEGMENT",
    "PREDICATED_UNSPSC_FAMILY",
    "PREDICATED_UNSPSC_CLASS",
    "PREDICATED_UNSPSC_COMMODITY",
    "PREDICATION_LASTUPDATED_AT"
]

PO_ITEM_COLUMNS = [
    "ID",
    "PO_ITEM_ID",
    "PROCESSED_DESCRIPTION",
    "EXTERNAL_PREDICATED_UNSPSC_SEGMENT",
    "EXTERNAL_PREDICATED_UNSPSC_FAMILY",
    "EXTERNAL_PREDICATED_UNSPSC_CLASS",
    "EXTERNAL_PREDICATED_UNSPSC_COMMODITY",
    "EXTERNAL_PREDICATION_LASTUPDATED_AT"
]


def _read_gds_report(source_table_location, target_columns, format):
    # Read the table data as a batch
    return sbnutils.get_spark() \
        .read.format(format) \
        .options(header='true', inferSchema='true') \
        .load(source_table_location) \
        .selectExpr(target_columns)


def _read_po_item(source_table_location, target_columns, format):
    return sbnutils.get_spark() \
        .read.format(format) \
        .options(header='true', inferSchema='true') \
        .load(source_table_location) \
        .selectExpr(target_columns) \
        .where(f"""EXTERNAL_PREDICATION_LASTUPDATED_AT is null""")


def _join_table(target_df, source_df, mode):
    return target_df.join(source_df, on=['PROCESSED_DESCRIPTION'], how=mode)


def _write_data(source_df, enrichment_zone, target_table_name):
    target_table = sbnutils.get_delta_table(enrichment_zone, target_table_name)
    cur_timestamp = current_timestamp()
    source_columns = source_df.columns

    update_expr = {f"target.{c}": f"source.{c}" for c in source_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp

    insert_expr = update_expr.copy()
    insert_expr[DELTA_CREATED_FIELD] = cur_timestamp
    if target_table_name.endswith(MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value):
        # Write the data to the delta table
        target_table.alias("target") \
            .merge(source_df.alias("source"), "source.ID = target.ID") \
            .whenMatchedUpdate(set=update_expr) \
            .execute()
    else:
        # Write the data to the delta table
        target_table.alias("target") \
            .merge(source_df.alias("source"), "source.PROCESSED_DESCRIPTION = target.PROCESSED_DESCRIPTION") \
            .whenMatchedUpdate(set=update_expr) \
            .whenNotMatchedInsert(values=insert_expr) \
            .execute()


def _filter_duplicated_data(po_item_gds_df):
    # Define a window function to partition by description and order by modification_time descending
    window = Window.partitionBy("PROCESSED_DESCRIPTION").orderBy(desc("PREDICATION_LASTUPDATED_AT"))
    # Assign a row number to each row within the window, filter to keep only the latest row for each description
    po_item_gds_df = po_item_gds_df.withColumn("row_number", row_number().over(window))
    po_item_gds_df = po_item_gds_df.filter(po_item_gds_df.row_number == 1)
    return po_item_gds_df.drop("row_number")


def _rename_po_item_gds_df(po_item_gds_df):
    return po_item_gds_df.withColumnRenamed("PREDICATED_UNSPSC_SEGMENT", "UNSPSC_SEGMENT") \
        .withColumnRenamed("PREDICATED_UNSPSC_FAMILY", "UNSPSC_FAMILY") \
        .withColumnRenamed("PREDICATED_UNSPSC_CLASS", "UNSPSC_CLASS") \
        .withColumnRenamed("PREDICATED_UNSPSC_COMMODITY", "UNSPSC_COMMODITY") \
        .withColumnRenamed("PREDICATION_LASTUPDATED_AT", "REPORT_LASTUPDATED_AT")


def _organize_ml_po_item_df(ml_po_item_df):
    df_po_item_updated = ml_po_item_df.withColumn("EXTERNAL_PREDICATED_UNSPSC_SEGMENT", col("UNSPSC_SEGMENT")) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_FAMILY", col("UNSPSC_FAMILY")) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_CLASS", col("UNSPSC_CLASS")) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_COMMODITY", col("UNSPSC_COMMODITY")) \
        .withColumn("EXTERNAL_PREDICATION_LASTUPDATED_AT", col("REPORT_LASTUPDATED_AT"))

    return df_po_item_updated.drop("UNSPSC_SEGMENT").drop("UNSPSC_FAMILY").drop("UNSPSC_CLASS").drop(
        "UNSPSC_COMMODITY") \
        .drop("CONFIDENCE_SEGMENT").drop("CONFIDENCE_FAMILY").drop("CONFIDENCE_CLASS").drop(
        "CONFIDENCE_COMMODITY").drop("REPORT_LASTUPDATED_AT")


def _archive_gds_csv(source_table_location):
    dbutils = sbnutils.get_dbutils()
    need_archive = dbutils.widgets.get("need_archive")
    sbnutils.log_info(f"Need archive: {need_archive}")

    if need_archive.lower() == "true":
        # Split the path into directory and filename parts
        dir_name, file_name = source_table_location.rsplit('/', 1)
        # Insert "archive" into the directory part of the path
        new_dir_name = dir_name + '/sap/gds_archive/'
        # Create archive location for GDS data
        archive_path = new_dir_name + file_name
        sbnutils.get_dbutils().fs.mkdirs(archive_path)
        sbnutils.log_info(archive_path)
        # Get a list of files in the origin path
        file_list = sbnutils.get_dbutils().fs.ls(source_table_location)
        # Loop through the file list and move each file to the archive path
        for file in file_list:
            if file.name.endswith('.csv'):
                sbnutils.get_dbutils().fs.mv(f'{file.path}', f'{archive_path}')
        sbnutils.get_dbutils().fs.rm(source_table_location, recurse=True)


def main():
    # Job Start
    sbnutils.log_info(f"Prediction read from GDS job Start")

    # Read Target table: ENRICHMENT_ML_SUPPLIER_PO_ITEM
    sbnutils.log_info(f"Load ml supplier data as dataframe")
    enrichment_zone = Zone.ENRICHMENT.value
    ml_po_item_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    ml_po_item_table_location = sbnutils.get_table_storage_location(enrichment_zone, ml_po_item_table_name)
    po_item_df = _read_po_item(ml_po_item_table_location, PO_ITEM_COLUMNS, 'delta')

    # Read the GDS prediction csv STAGING_ML_UNSPSC_REPORT_GDS as dataframe
    sbnutils.log_info(f"Load prediction from gds as dataframe")
    source_zone = Zone.STAGING.value
    source_table_name = MLTable.STAGING_ML_UNSPSC_REPORT_GDS.value
    source_table_location = sbnutils.get_table_storage_location(source_zone, source_table_name)
    po_item_gds_df = _read_gds_report(source_table_location, PO_ITEM_COLUMNS_EXTERNAL, 'csv')

    # Filter duplicated gds prediction with same description
    sbnutils.log_info(f"Move duplicated data and organize gds dataframe")
    po_item_gds_df = _filter_duplicated_data(po_item_gds_df)
    # Rename gds df to fit the gds report table
    po_item_gds_df = _rename_po_item_gds_df(po_item_gds_df)

    # merge UNSPSC to PO_ITEM table form GDS csv
    sbnutils.log_info(f"Merge gds df into ml supplier po item with new prediction")
    df_po_item_updated = _join_table(po_item_df, po_item_gds_df, 'inner')
    # Organize the UNSPSC updated df so that the updated df can merge into ml supplier table
    df_po_item_updated = _organize_ml_po_item_df(df_po_item_updated)

    # Write UNSPSC to ml_po_item table from GDS csv
    sbnutils.log_info(f"Write prediction into ml supplier po item delta table")
    _write_data(df_po_item_updated, enrichment_zone, ml_po_item_table_name)

    # update new UNSPSC to GDS table from GDS csv
    sbnutils.log_info(f"Write prediction into gds unspsc report delta table ends")
    gds_unspsc_report_table_name = MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value
    _write_data(po_item_gds_df, enrichment_zone, gds_unspsc_report_table_name)

    # archive the gds data
    sbnutils.log_info(f"Move prediction from gds to archive")
    _archive_gds_csv(source_table_location)

    # Job End
    sbnutils.log_info(f"Prediction read from GDS job End")
