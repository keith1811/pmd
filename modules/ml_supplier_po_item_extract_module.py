import sys
from pyspark.sql.functions import col, current_timestamp, lit, concat

from utils import sbnutils
from utils import batch_utils
from utils.constants import Zone, Table, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD

from modules.utils.constants import MLTable

PO_ITEM_COLUMNS = [
    "ID as PO_ITEM_ID",
    "PO as PO_ID",
    "DESCRIPTION",
    "MANUFACTURER_PART",
    "MANUFACTURER_NAME",
    "ITEM_TYPE",
    "ITEM_CATEGORY",
    "UNIT_OF_MEASURE",
    "SUPPLIER_PART",
    "SUPPLIER_PART_KID as SUPPLIER_PART_NUMBER",
    "SUPPLIER_PART_EXTENSION"
]

PO_ITEM_COMMODITY_COLUMNS = [
    "ID as PO_ITEM_COMMODITY_ID",
    "PO_ITEM",
    "DOMAIN",
    "CODE"
]

PO_COLUMNS = [
    "ID",
    "GENERIC_DOCUMENT as GENERIC_DOCUMENT_ID",
    "IS_BLANKET",
    "SHIP_TO_COUNTRY",
    "IS_ADHOC",
    "SERVICE_TYPE"
]

CXML_DOCUMENT_COLUMNS = [
    "ID as CXML_DOCUMENT_ID",
    "FROM_ORG as BUYER_ORG" ,
    "TO_ORG as SUPPLIER_ORG",
    "DASHBOARD_STATUS",
    "DOCUMENT_DATE",
    "CREATED as CREATED_DATE"
]

ORG_COLUMNS = [
    "ID as ORG_ID",
    "ANID",
    "NAME"
]


def _read_data(source_table_location, read_columns, where_condition = None):
    spark = sbnutils.get_spark()
    # Read the table data as a batch

    if where_condition:
        source_df = (
            spark.read.format("delta")
                .load(source_table_location)
                .where(where_condition)
                .selectExpr(read_columns)
        )
    else:
        source_df = (
            spark.read.format("delta")
                .load(source_table_location)
                .selectExpr(read_columns)
        )
    return source_df


def _write_data(source_df, target_table):
    cur_timestamp = current_timestamp()
    target_columns = target_table.toDF().columns

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


def _join_table(po_table, po_item_table, po_item_commodity_table, cxml_document_table, org_table):
    df_join = (
        po_item_table.alias("poi")
        .join(
            po_table.alias("po"),
            col("poi.PO_ID") == col("po.ID"),
            "inner"
        )
        .join(
            cxml_document_table.alias("cd"),
            col("po.GENERIC_DOCUMENT_ID") == col("cd.CXML_DOCUMENT_ID"),
            "inner"
        )
        .join(
            po_item_commodity_table.alias("pic"),
            col("poi.PO_ITEM_ID") == col("pic.PO_ITEM"),
            "left"
        )
        .join(
            org_table.alias("org"),
            col("BUYER_ORG") == col("org.ORG_ID"),
            "left"
        )
        .withColumnRenamed("NAME", "BUYER_NAME")
        .withColumnRenamed("ANID", "BUYER_ANID")
        .join(
            org_table.alias("org_1"),
            col("SUPPLIER_ORG") == col("org_1.ORG_ID"),
            "left"
        )
        .withColumnRenamed("NAME", "SUPPLIER_NAME")
        .withColumnRenamed("ANID", "SUPPLIER_ANID")
    )

    return df_join


def _extend_columns(df, target_table):
    # fill po_item_commodity na value
    df = df.na.fill({"PO_ITEM_COMMODITY_ID": "0"}) \
        .withColumn("ID", concat(col("PO_ITEM_ID"), lit("-"), col("PO_ITEM_COMMODITY_ID"))) \
        .withColumn("AN_UNSPSC_SEGMENT", lit(None)) \
        .withColumn("AN_UNSPSC_FAMILY", lit(None)) \
        .withColumn("AN_UNSPSC_CLASS", lit(None)) \
        .withColumnRenamed("CODE", "AN_UNSPSC_COMMODITY") \
        .withColumn("AN_DATA_QUALITY_LEVEL", lit(None)) \
        .withColumn("AN_DATA_QUALITY_LEVEL_MANUAL_REVIEWED", lit(None)) \
        .withColumn("AN_CLASSIFICATION_QUALITY_SEGMENT", lit(None)) \
        .withColumn("AN_CLASSIFICATION_QUALITY_FAMILY", lit(None)) \
        .withColumn("AN_CLASSIFICATION_QUALITY_CLASS", lit(None)) \
        .withColumn("AN_CLASSIFICATION_QUALITY_COMMODITY", lit(None)) \
        .withColumn("AN_CLASSIFICATION_QUALITY_LEVEL_MANUAL_REVIEWED", lit(None)) \
        .withColumn("AN_CLASSIFICATION_QUALITY_LEVEL_MANUAL_REVIEWED_AT", lit(None)) \
        .withColumn("SBN_PREDICATED_UNSPSC_SEGMENT", lit(None)) \
        .withColumn("SBN_PREDICATED_UNSPSC_FAMILY", lit(None)) \
        .withColumn("SBN_PREDICATED_UNSPSC_CLASS", lit(None)) \
        .withColumn("SBN_PREDICATED_UNSPSC_COMMODITY", lit(None)) \
        .withColumn("SBN_PREDICTION_CONFIDENCE_SEGMENT", lit(None)) \
        .withColumn("SBN_PREDICTION_CONFIDENCE_FAMILY", lit(None)) \
        .withColumn("SBN_PREDICTION_CONFIDENCE_CLASS", lit(None)) \
        .withColumn("SBN_PREDICTION_CONFIDENCE_COMMODITY", lit(None)) \
        .withColumn("SBN_PREDICTION_LASTUPDATED_AT", lit(None)) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_SEGMENT", lit(None)) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_FAMILY", lit(None)) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_CLASS", lit(None)) \
        .withColumn("EXTERNAL_PREDICATED_UNSPSC_COMMODITY", lit(None)) \
        .withColumn("EXTERNAL_PREDICATION_CONFIDENCE_SEGMENT", lit(None)) \
        .withColumn("EXTERNAL_PREDICATION_CONFIDENCE_FAMILY", lit(None)) \
        .withColumn("EXTERNAL_PREDICATION_CONFIDENCE_CLASS", lit(None)) \
        .withColumn("EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY", lit(None)) \
        .withColumn("EXTERNAL_PREDICATION_LASTUPDATED_AT", lit(None)) \
        .withColumn("FINAL_REPORT_UNSPSC_SEGMENT", lit(None)) \
        .withColumn("FINAL_REPORT_UNSPSC_FAMILY", lit(None)) \
        .withColumn("FINAL_REPORT_UNSPSC_CLASS", lit(None)) \
        .withColumn("FINAL_REPORT_UNSPSC_COMMODITY", lit(None)) \
        .withColumn("FINAL_REPORT_CONFIDENCE_SEGMENT", lit(None)) \
        .withColumn("FINAL_REPORT_CONFIDENCE_FAMILY", lit(None)) \
        .withColumn("FINAL_REPORT_CONFIDENCE_CLASS", lit(None)) \
        .withColumn("FINAL_REPORT_CONFIDENCE_COMMODITY", lit(None)) \
        .withColumn("FINAL_REPORT_UNSPSC_MANUAL_LABELED", lit(None)) \
        .withColumn("FINAL_REPORT_LASTUPDATED_AT", lit(None)) \
        .withColumn("FINAL_REPORT_SOURCE", lit(None)) \
        .withColumn("PROCESSED_DESCRIPTION", lit(None)) \
        .withColumn("BANNED_FROM_TRAINING", lit(None)) \
        .withColumn("BANNED_FROM_ENRICHMENT", lit(None)) \
        .withColumn("MODEL_UUID", lit(None)) \
        .selectExpr(_get_target_table_columns(target_table)) \
        .withColumn(DELTA_CREATED_FIELD, current_timestamp()) \
        .withColumn(DELTA_UPDATED_FIELD, current_timestamp()) \
        .dropDuplicates(["ID"])
    return df


def _get_target_table_columns(target_table):
    final_columns = target_table.toDF().columns
    final_columns.remove(DELTA_UPDATED_FIELD)
    final_columns.remove(DELTA_CREATED_FIELD)
    return final_columns


def main():
    timestamp_range = batch_utils.get_batch_timestamp_range()
    sbnutils.log_info(f"Batch starting time: {timestamp_range[0]}")
    sbnutils.log_info(f"Batch ending time: {timestamp_range[1]}")

    source_zone = Zone.RAW.value
    po_table_name = Table.RAW_PO.value
    po_item_table_name = Table.RAW_PO_ITEM.value
    po_item_commodity_table_name = Table.RAW_PO_ITEM_COMMODITY.value
    cxml_document_table_name = Table.RAW_CXML_DOCUMENT.value
    org_table_name = Table.RAW_ORG.value

    # Target table
    target_zone = Zone.ENRICHMENT.value
    target_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    target_table = sbnutils.get_delta_table(target_zone, target_table_name)

    # Source table location
    po_table_location = sbnutils.get_table_storage_location(source_zone, po_table_name)
    po_item_table_location = sbnutils.get_table_storage_location(source_zone, po_item_table_name)
    po_item_commodity_table_location = sbnutils.get_table_storage_location(source_zone, po_item_commodity_table_name)
    cxml_document_table_location = sbnutils.get_table_storage_location(source_zone, cxml_document_table_name)
    org_table_location = sbnutils.get_table_storage_location(source_zone, org_table_name)

    # Job Start
    sbnutils.log_info(f"ML supplier po item enrichment batch job Start")
    sbnutils.log_info(
        f"is_integration_test_mode: {sbnutils._is_integration_test_mode()}"
    )

    timestamp_range_condition = f"""
            {DELTA_UPDATED_FIELD} >= '{timestamp_range[0]}' AND
            {DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}'
        """

    # po maintain obsolete data, fetch delta records
    sbnutils.log_info(f"Load source table to dataframe")
    df_po = _read_data(po_table_location, PO_COLUMNS, timestamp_range_condition + " AND IS_BLANKET == 0")

    # po_item not maintain obsolete data, need to fetch all records
    df_po_item = _read_data(po_item_table_location, PO_ITEM_COLUMNS)

    df_po_item_commodity = _read_data(po_item_commodity_table_location, PO_ITEM_COMMODITY_COLUMNS)

    # cxml_document maintain obsolete data, fetch delta records
    df_cxml_document = _read_data(cxml_document_table_location, CXML_DOCUMENT_COLUMNS, timestamp_range_condition)

    df_org = _read_data(org_table_location, ORG_COLUMNS)

    sbnutils.log_info(f"Join tables into joined dataframe")
    df_joined = _join_table(po_table=df_po, \
                            po_item_table=df_po_item, \
                            po_item_commodity_table= df_po_item_commodity, \
                            cxml_document_table=df_cxml_document, \
                            org_table=df_org)

    sbnutils.log_info(f"Extend columns and add db updated filed")
    df_extend = _extend_columns(df_joined, target_table)

    _write_data(df_extend, target_table)
    sbnutils.log_info(f"ML supplier po item enrichment batch job End")