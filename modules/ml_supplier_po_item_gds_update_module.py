from modules.utils.constants import MLTable
from utils import sbnutils
from utils import constants
from utils import batch_utils
from utils.constants import DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from utils.constants import Zone
from pyspark.sql.functions import current_timestamp, col

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


def main():
    # Job Start
    sbnutils.log_info(f"Prediction update from unspsc report job starts")

    # range read from Target table: ENRICHMENT_ML_SUPPLIER_PO_ITEM
    enrichment_zone = Zone.ENRICHMENT.value
    ml_po_item_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    ml_po_item_table_location = sbnutils.get_table_storage_location(enrichment_zone, ml_po_item_table_name)
    po_item_df = _read_data(ml_po_item_table_location, PO_ITEM_COLUMNS, 'delta')

    # get source table ENRICHMENT_ML_GDS_UNSPSC_REPORT unspsc code
    ml_gds_unspsc_table_name = MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value
    sbnutils.log_info(f"Load prediction from unspsc report starts")
    gds_unspsc_table = sbnutils.get_delta_table(enrichment_zone, ml_gds_unspsc_table_name)
    sbnutils.log_info(f"Load prediction from unspsc report ends")

    # check for null values in column EXTERNAL_PREDICATED_UNSPSC_SEGMENT_OLD from ENRICHMENT_ML_SUPPLIER_PO_ITEM
    sbnutils.log_info(f"Updating GDS data")
    po_item_df = _join_table(po_item_df, gds_unspsc_table.toDF(), "inner")

    # remove extra columns which shouldn't insert into ENRICHMENT_ML_SUPPLIER_PO_ITEM
    po_item_df = _format_data(po_item_df)
    sbnutils.log_info(f"Write prediction into delta tab starts")

    # Write UNSPSC to ENRICHMENT_ML_SUPPLIER_PO_ITEM from ENRICHMENT_ML_GDS_UNSPSC_REPORT
    ml_po_item_table = sbnutils.get_delta_table(enrichment_zone, ml_po_item_table_name)
    _write_data(po_item_df, ml_po_item_table)


def _read_data(source_table_location, target_columns, format):
    # Read the table data as a batch
    timestamp_range = batch_utils.get_batch_timestamp_range()
    return sbnutils.get_spark() \
        .read.format(format) \
        .options(header='true', inferSchema='true') \
        .load(source_table_location) \
        .selectExpr(target_columns) \
        .where(f"""
            {constants.DELTA_UPDATED_FIELD} >= '{timestamp_range[0]}' AND 
            {constants.DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}' AND
            EXTERNAL_PREDICATION_LASTUPDATED_AT is null
            """)


def _join_table(target_df, source_df, mode):
    return target_df.join(source_df, on=['PROCESSED_DESCRIPTION'], how=mode)


def _write_data(source_df, target_table):
    cur_timestamp = current_timestamp()
    source_columns = source_df.columns

    update_expr = {f"target.{c}": f"source.{c}" for c in source_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp

    # Write the data to the delta table
    target_table.alias("target") \
        .merge(source_df.alias("source"),
               "source.ID = target.ID AND (target.EXTERNAL_PREDICATION_LASTUPDATED_AT IS NULL OR \
               source.EXTERNAL_PREDICATION_LASTUPDATED_AT != target.EXTERNAL_PREDICATION_LASTUPDATED_AT)") \
        .whenMatchedUpdate(set=update_expr) \
        .execute()


def _format_data(po_item_df):
    po_item_df = po_item_df.withColumn("EXTERNAL_PREDICATED_UNSPSC_SEGMENT", col("UNSPSC_SEGMENT")) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_FAMILY", col("UNSPSC_FAMILY")) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_CLASS", col("UNSPSC_CLASS")) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_COMMODITY", col("UNSPSC_COMMODITY")) \
        .withColumn("EXTERNAL_PREDICATION_LASTUPDATED_AT", col("REPORT_LASTUPDATED_AT"))

    return po_item_df.drop("UNSPSC_SEGMENT").drop("UNSPSC_FAMILY").drop("UNSPSC_CLASS").drop("UNSPSC_COMMODITY") \
        .drop("CONFIDENCE_SEGMENT").drop("CONFIDENCE_FAMILY").drop("CONFIDENCE_CLASS").drop("CONFIDENCE_COMMODITY") \
        .drop("REPORT_LASTUPDATED_AT").drop("_DELTA_UPDATED_ON").drop("_DELTA_CREATED_ON")
