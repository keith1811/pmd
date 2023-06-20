from utils import sbnutils
from utils.constants import Zone, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from modules.utils.constants import MLTable
from pyspark.sql.functions import current_timestamp
import decimal
# Copy buyer_po_item from raw zone to training zone(manually, full replication);

def main():
    # Job Start
    sbnutils.log_info("Replication job Start")

    source_zone = Zone.RAW.value
    source_table_name = MLTable.RAW_ML_BUYER_PO_ITEM.value
    source_table_location = sbnutils.get_table_storage_location(source_zone, source_table_name)

    target_zone = "training"
    target_table_name = MLTable.TRAINING_ML_BUYER_PO_ITEM.value
    target_table = sbnutils.get_delta_table(target_zone, target_table_name)

    # Read buyer data from raw zone
    buyer_po_item = _read_data(source_table_location)

    #sbnutils.log_info(source_table_location)

    # Write to training zone
    _write_data(buyer_po_item, target_table)

    sbnutils.log_info("Replication job End")

def _read_data(source_table_location):

    spark = sbnutils.get_spark()
    source_df = (
        spark.read.format("delta").load(source_table_location)
    )
    return source_df

def _write_data(source_df, target_table):
    sbnutils.log_info(f"Writing data")
    cur_timestamp = current_timestamp()

    target_columns = target_table.toDF().columns
    sbnutils.log_info(target_columns)

    # Update
    target_columns.remove(DELTA_UPDATED_FIELD)
    update_expr = {f"target.{c}": f"source.{c}" for c in target_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp

    # Insert
    target_columns.remove(DELTA_CREATED_FIELD)
    insert_expr = {f"target.{c}": f"source.{c}" for c in target_columns}
    insert_expr[DELTA_CREATED_FIELD] = cur_timestamp
    insert_expr[DELTA_UPDATED_FIELD] = cur_timestamp

    # Write the data to the delta table
    target_table.alias("target") \
        .merge(source_df.alias("source"),
               "source.ID = target.ID") \
        .whenMatchedUpdate(set=update_expr) \
        .whenNotMatchedInsert(values=insert_expr) \
        .execute()
