import sys
from pyspark.sql.functions import current_timestamp, when, isnull, concat, year, lpad, month, col, avg, countDistinct, \
    sum, lit, trunc
from pyspark.sql.types import StringType, IntegerType
from utils.constants import Zone, Table, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from utils import sbnutils
from utils import batch_utils

from modules.utils.constants import MLTable

def main():
    source_zone = Zone.CONSUMPTION.value
    source_table_name = MLTable.CONSUMPTION_FACT_SUPPLIER_PO_ITEM.value
    dim_table_name = MLTable.CONSUMPTION_DIM_ML_SUPPLIER_PO_ITEM.value
    target_zone = Zone.CONSUMPTION.value
    target_table_name = MLTable.CONSUMPTION_NETWORK_SUPPLIER_PO_ITEM.value
    source_table_location = sbnutils.get_table_storage_location(
        source_zone, source_table_name
    )
    dim_table_location = sbnutils.get_table_storage_location(
        source_zone, dim_table_name
    )

    ## Target table
    target_table = sbnutils.get_delta_table(target_zone, target_table_name)

    source_df = _read_data(source_table_location)

    dim_df = _read_dimension_data(dim_table_location)

    source_df = _aggregate_data(source_df, dim_df)

    _write_data(source_df, target_table)

    sbnutils.log_info("Aggregation fact supplier po item job ends")

def _read_data(source_table_location):
    spark = sbnutils.get_spark()
    timestamp_range = batch_utils.get_batch_timestamp_range()

    # Read the table data as a batch
    source_df = (
        spark.read.format("delta")
        .load(source_table_location)
        .where(
            f"""
            DOCUMENT_DATE >= '{timestamp_range[0]}' AND 
            DOCUMENT_DATE <= '{timestamp_range[1]}' AND
            IS_BLANKET == 0
            """
        )
    )

    return source_df

def _read_dimension_data(dim_table_location):
    spark = sbnutils.get_spark()
    timestamp_range = batch_utils.get_batch_timestamp_range()

    # Read the table data as a batch
    dim_df = (
        spark.read.format("delta")
        .load(dim_table_location)
    )

    return dim_df

def _write_data(source_df, target_table):
    # Write the stream to the delta table
    cur_timestamp = current_timestamp()
    target_columns = target_table.toDF().columns
    target_columns.remove(DELTA_UPDATED_FIELD)
    target_columns.remove(DELTA_CREATED_FIELD)
    update_expr = {f"target.{c}": f"source.{c}" for c in target_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp
    sbnutils.log_info(f"Writing data")
    (
        target_table.alias("target")
        .merge(source_df.alias("source"), "source.SUPPLIER_ANID = target.SUPPLIER_ANID AND source.BUYER_ORG = target.BUYER_ORG AND source.SUPPLIER_ORG = target.SUPPLIER_ORG AND source.REQUESTED_DELIVERY_DATE = target.REQUESTED_DELIVERY_DATE AND source.DOCUMENT_DATE = target.DOCUMENT_DATE AND source.SHIP_TO_COUNTRY = target.SHIP_TO_COUNTRY AND source.UNIT_OF_MEASURE = target.UNIT_OF_MEASURE AND source.UNSPSC_COMMODITY = target.UNSPSC_COMMODITY")
        .whenMatchedUpdate(set=update_expr)
        .whenNotMatchedInsertAll()
        .execute()
    )

def _aggregate_data(source_df, dim_df):
    cur_timestamp = current_timestamp()
    source_df = source_df.join(dim_df, source_df.PRODUCT_ID == dim_df.ID, "left").drop(dim_df.ID)
    source_df = source_df.na.fill("99999999", ["UNSPSC_COMMODITY"])
    source_df = source_df.na.fill("999999", ["UNSPSC_CLASS"])
    source_df = source_df.na.fill("9999", ["UNSPSC_FAMILY"])
    source_df = source_df.na.fill("99", ["UNSPSC_SEGMENT"])
    source_df = source_df.withColumn("REQUESTED_DELIVERY_DATE", trunc("REQUESTED_DELIVERY_DATE", "MM"))
    source_df = source_df.withColumn("DOCUMENT_DATE", trunc("DOCUMENT_DATE", "MM"))
    source_df = source_df.withColumn("OA", (col("QUANTITY") * col("UNIT_PRICE_USD")).cast(IntegerType()))
    source_df = source_df.groupBy("SUPPLIER_ANID", "BUYER_ORG", "SUPPLIER_ORG", "REQUESTED_DELIVERY_DATE", "DOCUMENT_DATE", "SHIP_TO_COUNTRY", "UNIT_OF_MEASURE", "UNSPSC_SEGMENT", "UNSPSC_FAMILY", "UNSPSC_CLASS", "UNSPSC_COMMODITY")\
                         .agg(sum("OA").alias("ORDER_AMOUNT"),
                              sum("QUANTITY").alias("ORDER_UNITS"),
                              avg("UNIT_PRICE_USD").alias("AVERAGE_UNIT_PRICE_USD"),
                              countDistinct("PO").alias("PO_COUNT"),
                              countDistinct("ID").alias("PO_ITEM_COUNT"))
    source_df = source_df.withColumn(DELTA_CREATED_FIELD, lit(cur_timestamp))
    source_df = source_df.withColumn(DELTA_UPDATED_FIELD, lit(cur_timestamp))
    return source_df