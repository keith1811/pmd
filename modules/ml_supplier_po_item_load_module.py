from utils import sbnutils, batch_utils
from utils.constants import Zone, Table, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from modules.utils.constants import MLTable
from pyspark.sql.functions import current_timestamp
import datetime
import pandas as pd

# The loading job have the following open issues, need to be enhanced in the future:
# 1. Can we remove the condition of filtering out PO_OBSOLETE data?
# 2. We use the dropDuplicates function to random drop duplicates, how to make it stable?
# 3. To avoid the fact table late arrival issue, extend the timerange to a bigger range here, it can be enhanced.
# 4. Reprocessing issue. If the fact table add 2019's data, our table need to trigger the earlier data processing.

# Case 1: id 1-3 are new data for all tables
# ml_supplier_po_item
# ID    PO_ITEM_ID  DASHBOARD_STATUS    FINAL_COMMODITY
# 1     10001       PO_NEW              11111111
# 2     10001       PO_NEW              11111111
# 3     10001       PO_NEW              11111111

# before merge into dim table,
# 1. filter out DASHBOARD_STATUS == PO_OBSOLETE data
# 2. drop duplicate po_item_id

# dim
# ID    PO_ITEM_ID  FINAL_COMMODITY   createdon   updatedon(更新到未来的时间)
# 1     10001       11111111

# merge condition: dim.PO_ITEM_ID == fact.ID and (fact.PRODUCT_ID is null or fact.PRODUCT_ID != dim.ID)
# fact
# ID    PRODUCT_ID
# 10001 1

# Case 2: ID 1-3 update the final commodity, will not insert new columns, dbupdated will change.
# ml_supplier_po_item
# ID    PO_ITEM_ID  DASHBOARD_STATUS    FINAL_COMMODITY
# 1     10001       PO_NEW              22222222
# 2     10001       PO_NEW              22222222
# 3     10001       PO_NEW              22222222

# after filter out po_obsolete and drop duplicate, will fetch one record with updated final commodity
# dim
# ID    PO_ITEM_ID  FINAL_COMMODITY
# 1     10001       11111111
# 2     10001       22222222

# merge condition: dim.PO_ITEM_ID == fact.ID and (fact.PRODUCT_ID is null or fact.PRODUCT_ID != dim.ID)
# fact
# ID    PRODUCT_ID
# 10001 2

# case 3: id 1-3 obsolete, a new record will be inserted
# ID    PO_ITEM_ID  DASHBOARD_STATUS    FINAL_COMMODITY
# 1     10001       PO_OBSOLETE         22222222
# 2     10001       PO_OBSOLETE         22222222
# 3     10001       PO_OBSOLETE         22222222
# 4     10001-2     PO_NEW              11111112

# filter out obsolete and drop duplicate, 10001 won't be updated. But a new record will be inserted.
# dim
# ID    PO_ITEM_ID  FINAL_COMMODITY
# 1     10001       11111111   # won't be update
# 4     10001-2     11111112

# fact
# ID    PRODUCT_ID
# 10001     1
# 10001-2   4

ML_SUPPLIER_PO_ITEM_COLUMNS = [
    "ID",
    "PO_ITEM_ID",
    "DESCRIPTION",
    "DASHBOARD_STATUS",
    "FINAL_REPORT_UNSPSC_SEGMENT as UNSPSC_SEGMENT",
    "FINAL_REPORT_UNSPSC_FAMILY as UNSPSC_FAMILY",
    "FINAL_REPORT_UNSPSC_CLASS as UNSPSC_CLASS",
    "FINAL_REPORT_UNSPSC_COMMODITY as UNSPSC_COMMODITY",
    "FINAL_REPORT_CONFIDENCE_SEGMENT as UNSPSC_CONFIDENCE_SEGMENT",
    "FINAL_REPORT_CONFIDENCE_FAMILY as UNSPSC_CONFIDENCE_FAMILY",
    "FINAL_REPORT_CONFIDENCE_CLASS as UNSPSC_CONFIDENCE_CLASS",
    "FINAL_REPORT_CONFIDENCE_COMMODITY as UNSPSC_CONFIDENCE_COMMODITY",
    "FINAL_REPORT_LASTUPDATED_AT as REPORT_LASTUPDATED_AT",
    "FINAL_REPORT_SOURCE as REPORT_SOURCE"
]


def _read_data(source_table_location, read_columns, where_condition = "1 == 1"):
    spark = sbnutils.get_spark()

    source_df = (
        spark.read.format("delta")
            .load(source_table_location)
            .where(where_condition)
            .selectExpr(read_columns)
    )
    return source_df

def _write_data_to_dim(source_df, target_table):
    cur_timestamp = current_timestamp()
    target_columns = target_table.toDF().columns # dim columns
    target_columns.remove(DELTA_UPDATED_FIELD)
    target_columns.remove(DELTA_CREATED_FIELD)

    update_expr = {f"target.{c}": f"source.{c}" for c in target_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp

    insert_expr = update_expr.copy()
    insert_expr[DELTA_CREATED_FIELD] = cur_timestamp

    # Write the data to the delta table
    (
        target_table.alias("target")
            .merge(source_df.alias("source"), "source.ID = target.ID")
            .whenMatchedUpdate(set=update_expr)
            .whenNotMatchedInsert(values=insert_expr)
            .execute()
    )


def _write_data_to_fact(source_df, target_table):
    cur_timestamp = current_timestamp()
    update_expr = {"target.PRODUCT_ID": "source.ID"}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp

    # Write the data to the delta table
    (
        target_table.alias("target")
            .merge(source_df.alias("source"), "source.PO_ITEM_ID = target.ID \
            and target.IS_BLANKET == 0 \
            and (target.PRODUCT_ID IS NULL or source.ID != target.PRODUCT_ID)") # when fact PRODUCT_ID is NULL, the source.ID != target.PRODUCT_ID won't be established
            .whenMatchedUpdate(set=update_expr)
            .execute()
    )


def main():
    timestamp_range = batch_utils.get_batch_timestamp_range()
    sbnutils.log_info(f"Batch starting time: {timestamp_range[0]}")
    sbnutils.log_info(f"Batch ending time: {timestamp_range[1]}")

    ml_supplier_po_item_table_location = sbnutils.get_table_storage_location(Zone.ENRICHMENT.value, Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)

    dim_ml_supplier_po_item_table = sbnutils.get_delta_table(Zone.CONSUMPTION.value, MLTable.CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM.value)
    fact_supplier_po_item_table = sbnutils.get_delta_table(Zone.CONSUMPTION.value, Table.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value)

    # Job Start
    sbnutils.log_info("ML supplier po item load batch job Start")
    sbnutils.log_info(f"is_integration_test_mode: {sbnutils._is_integration_test_mode()}")

    starting_time =  pd.to_datetime(timestamp_range[0]) - datetime.timedelta(days=7)
    sbnutils.log_info(f"Time range: {timestamp_range[0]}, extended starting time: {starting_time}")

    timestamp_range_condition = f"""
            {DELTA_UPDATED_FIELD} >= '{starting_time}' AND
            {DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}'
        """

    sbnutils.log_info("Load ml po item table to dataframe")
    # read increment data of ml_supplier_po_item with extend time range
    ml_po_item_df = _read_data(ml_supplier_po_item_table_location, ML_SUPPLIER_PO_ITEM_COLUMNS, timestamp_range_condition)
    ml_po_item_df = ml_po_item_df.where("DASHBOARD_STATUS != 'PO_OBSOLETE'").dropDuplicates(["PO_ITEM_ID"])

    sbnutils.log_info("Write to dimension table")
    _write_data_to_dim(ml_po_item_df, dim_ml_supplier_po_item_table)

    sbnutils.log_info("Write to fact table")
    _write_data_to_fact(ml_po_item_df, fact_supplier_po_item_table)

    sbnutils.log_info("ML supplier po item load batch job End")