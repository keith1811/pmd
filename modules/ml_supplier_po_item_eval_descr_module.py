import sys

from pyspark.sql.functions import rtrim, ltrim, array_distinct, concat_ws, split, lower, regexp_replace, when, length, \
    col, current_timestamp
from pyspark.sql.types import StringType

from utils import sbnutils
from utils import batch_utils
from utils.constants import Zone, Table, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD

from modules.utils.constants import MLTable

def main():
    # Job Start
    sbnutils.log_info(f"Processing batch job Start")

    source_zone = Zone.ENRICHMENT.value
    source_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    target_zone = Zone.ENRICHMENT.value
    target_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    commodity_table_name = "commodity"

    ## Source table
    source_table_location = sbnutils.get_table_storage_location(
        source_zone, source_table_name
    )

    ## Target table
    target_table = sbnutils.get_delta_table(target_zone, target_table_name)

    # Commodity Table
    commodity_table_location = sbnutils.get_table_storage_location(
        Zone.RAW.value, commodity_table_name
    )

    # Read the whole commodity table to a df
    commodity_df = _read_commodity(commodity_table_location)

    # Read the source table data as a batch
    source_df = _read_data(source_table_location)

    # Preprocess the data to remove punctuation and to lower case and remove duplicated words
    source_df = _clean_data(source_df)
    
    # Evaluate preprocessed description
    source_df = _evaluate_description(source_df)
    
    # Check if UNSPSC code is valid
    source_df = _evaluate_unspsc(source_df, commodity_df)
    
    # Derive UNSPSC code to four levels to four columns
    source_df = _derive_unspsc(source_df)

    # Write the batch to the the target table
    _write_data(source_df, target_table)

    # Job End
    sbnutils.log_info(f"Sample batch job End")


def _read_data(source_table_location):
    spark = sbnutils.get_spark()
    timestamp_range = batch_utils.get_batch_timestamp_range()

    # Read the table data as a batch
    source_df = (
        spark.read.format("delta")
        .load(source_table_location)
        .where(
            f"""
            {DELTA_UPDATED_FIELD} >= '{timestamp_range[0]}' AND 
            {DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}'
            """
        )
    )

    return source_df


def _read_commodity(commodity_table_location):
    spark = sbnutils.get_spark()

    # Read the table data as a batch
    source_df = (
        spark.read.format("delta")
        .load(commodity_table_location)
        .select("CODE")
        .where(
            f"""
            VALID == 1
            """
        )
    )

    return source_df


def _write_data(source_df, target_table):
    # Write the stream to the delta table
    cur_timestamp = current_timestamp()
    target_columns = target_table.toDF().columns
    target_columns.remove(DELTA_UPDATED_FIELD)
    target_columns.remove(DELTA_CREATED_FIELD)

    update_expr = {f"t.{c}": f"s.{c}" for c in target_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp
    sbnutils.log_info(f"Writing data")
    (
        target_table.alias("t")
        .merge(source_df.alias("s"), "s.ID = t.ID AND (s.PROCESSED_DESCRIPTION != t.PROCESSED_DESCRIPTION \
            OR t.AN_CLASSIFICATION_QUALITY_COMMODITY is NULL \
            OR s.AN_CLASSIFICATION_QUALITY_COMMODITY != t.AN_CLASSIFICATION_QUALITY_COMMODITY \
            OR s.AN_UNSPSC_COMMODITY != t.AN_UNSPSC_COMMODITY)")
        .whenMatchedUpdate(set=update_expr)
        .execute()
    )


def _clean_data(source_df):
    source_df = source_df \
        .withColumn("PROCESSED_DESCRIPTION", rtrim(ltrim(
        concat_ws(" ", array_distinct(split(lower(regexp_replace("DESCRIPTION", r"[!\"#$%&'()*+,\-.\/:;<=>?@\[\\\]^_`{|}~]", " ")), " "))))))
    return source_df


def _evaluate_description(source_df):
    source_df = source_df.withColumn("AN_DATA_QUALITY_LEVEL",
                                     (when((length(regexp_replace('PROCESSED_DESCRIPTION', '[^0-9a-z]', '')) < 3) |
                                           (length(regexp_replace('PROCESSED_DESCRIPTION', '[^a-z]', '')) < 3) |
                                           (length('PROCESSED_DESCRIPTION') < 3), 'Poor')
                                      .when((length(regexp_replace('PROCESSED_DESCRIPTION', '[^0-9a-z]', '')) > 15) &
                                            (length(regexp_replace('PROCESSED_DESCRIPTION', '[^a-z]', '')) > 15) &
                                            (length('PROCESSED_DESCRIPTION') > 15), 'Good')
                                      .otherwise('Acceptable')))
    return source_df


def _evaluate_unspsc(source_df, commodity_df):
    source_df = source_df.na.fill("0", ["AN_UNSPSC_COMMODITY"])
    source_df = source_df.withColumn("AN_UNSPSC_COMMODITY",
                                     regexp_replace("AN_UNSPSC_COMMODITY", r"0{2}\b|0{4}\b|0{6}\b", ""))
    commodity_df = commodity_df.withColumn("CODE", commodity_df.CODE.cast(StringType()))
    source_df = source_df.join(commodity_df, source_df.AN_UNSPSC_COMMODITY == commodity_df.CODE, "left")
    source_df = source_df.withColumn("QUALITY", when(col("CODE").isNull(), "Poor").otherwise("Acceptable"))
    source_df = source_df.withColumn("AN_CLASSIFICATION_QUALITY_SEGMENT", source_df.QUALITY)
    source_df = source_df.withColumn("AN_CLASSIFICATION_QUALITY_FAMILY", source_df.QUALITY)
    source_df = source_df.withColumn("AN_CLASSIFICATION_QUALITY_CLASS", source_df.QUALITY)
    source_df = source_df.withColumn("AN_CLASSIFICATION_QUALITY_COMMODITY", source_df.QUALITY)
    source_df = source_df.drop("CODE", "QUALITY")
    return source_df


def _derive_unspsc(source_df):
    source_df = source_df.withColumn("AN_UNSPSC_SEGMENT", 
                                     when(source_df.AN_CLASSIFICATION_QUALITY_COMMODITY == "Acceptable", 
                                          source_df.AN_UNSPSC_COMMODITY[0:2]).otherwise("0"))
    source_df = source_df.withColumn("AN_UNSPSC_FAMILY", 
                                     when(source_df.AN_CLASSIFICATION_QUALITY_COMMODITY == "Acceptable", 
                                          source_df.AN_UNSPSC_COMMODITY[0:4]).otherwise("0"))
    source_df = source_df.withColumn("AN_UNSPSC_CLASS", 
                                     when(source_df.AN_CLASSIFICATION_QUALITY_COMMODITY == "Acceptable", 
                                          source_df.AN_UNSPSC_COMMODITY[0:6]).otherwise("0"))
    source_df = source_df.withColumn("AN_UNSPSC_COMMODITY",
                                     when(source_df.AN_CLASSIFICATION_QUALITY_COMMODITY == "Acceptable",
                                          source_df.AN_UNSPSC_COMMODITY[0:8]).otherwise("0"))
    return source_df
