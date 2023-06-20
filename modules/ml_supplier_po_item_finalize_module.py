from pyspark.sql.functions import current_timestamp, when

from utils import sbnutils
from utils import batch_utils
from utils.constants import Zone, Table, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from modules.utils.notebook_utils import load_and_set_env_vars

from modules.utils.constants import MLTable
def main():
    # Job Start
    sbnutils.log_info(f"Processing batch job Start")

    source_zone = Zone.ENRICHMENT.value
    source_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
    target_zone = Zone.ENRICHMENT.value
    target_table_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value

    source_table_location = sbnutils.get_table_storage_location(
        source_zone, source_table_name
    )

    target_table = sbnutils.get_delta_table(target_zone, target_table_name)

    source_df = _read_data(source_table_location)

    source_df = _finalize_unspsc_commodity(source_df)

    source_df = _derive_unspsc(source_df)

    _write_data(source_df, target_table)

    sbnutils.log_info(f"po item finalize job End")

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
            {DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}' AND
            FINAL_REPORT_UNSPSC_MANUAL_LABELED is null
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

    update_expr = {f"target.{c}": f"source.{c}" for c in target_columns}
    update_expr[DELTA_UPDATED_FIELD] = cur_timestamp
    sbnutils.log_info(f"Writing data")
    (
        target_table.alias("target")
        .merge(source_df.alias("source"), "source.ID = target.ID")
        .whenMatchedUpdate(set=update_expr)
        .whenNotMatchedInsertAll()
        .execute()
    )

def _finalize_unspsc_commodity(source_df):
    env = sbnutils._get_env()
    env_vars = load_and_set_env_vars(env=env)
    min_confidence = env_vars['min_confidence']
    source_df = source_df.withColumn("FINAL_REPORT_UNSPSC_COMMODITY",
                                     when(source_df.EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY > min_confidence, source_df.EXTERNAL_PREDICATED_UNSPSC_COMMODITY)
                                     .when(source_df.SBN_PREDICTION_CONFIDENCE_COMMODITY > min_confidence, source_df.SBN_PREDICATED_UNSPSC_COMMODITY)
                                     .when(source_df.AN_CLASSIFICATION_QUALITY_COMMODITY == "Good", source_df.AN_UNSPSC_COMMODITY)
                                     .when((source_df.AN_UNSPSC_COMMODITY == source_df.EXTERNAL_PREDICATED_UNSPSC_COMMODITY) | (source_df.AN_UNSPSC_COMMODITY == source_df.SBN_PREDICATED_UNSPSC_COMMODITY), source_df.AN_UNSPSC_COMMODITY)
                                     )
    source_df = source_df.withColumn("FINAL_REPORT_CONFIDENCE_COMMODITY",
                                     when(source_df.EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY > min_confidence,
                                          source_df.EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY)
                                     .when(source_df.SBN_PREDICTION_CONFIDENCE_COMMODITY > min_confidence,
                                           source_df.SBN_PREDICTION_CONFIDENCE_COMMODITY)
                                     .when(source_df.AN_UNSPSC_COMMODITY == source_df.EXTERNAL_PREDICATED_UNSPSC_COMMODITY,
                                         source_df.EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY)
                                     .when(source_df.AN_UNSPSC_COMMODITY == source_df.SBN_PREDICATED_UNSPSC_COMMODITY,
                                           source_df.SBN_PREDICTION_CONFIDENCE_COMMODITY)
                                     )
    source_df = source_df.withColumn("FINAL_REPORT_CONFIDENCE_SEGMENT", source_df.FINAL_REPORT_CONFIDENCE_COMMODITY)
    source_df = source_df.withColumn("FINAL_REPORT_CONFIDENCE_FAMILY", source_df.FINAL_REPORT_CONFIDENCE_COMMODITY)
    source_df = source_df.withColumn("FINAL_REPORT_CONFIDENCE_CLASS", source_df.FINAL_REPORT_CONFIDENCE_COMMODITY)
    cur_timestamp = current_timestamp()
    source_df = source_df.withColumn("FINAL_REPORT_LASTUPDATED_AT", when(source_df.FINAL_REPORT_CONFIDENCE_COMMODITY.isNotNull(), cur_timestamp))
    return source_df

def _derive_unspsc(source_df):
    source_df = source_df.withColumn("FINAL_REPORT_UNSPSC_SEGMENT",
                                     when(source_df.FINAL_REPORT_UNSPSC_COMMODITY.isNotNull(),
                                          source_df.FINAL_REPORT_UNSPSC_COMMODITY[0:2]))
    source_df = source_df.withColumn("FINAL_REPORT_UNSPSC_FAMILY",
                                     when(source_df.FINAL_REPORT_UNSPSC_COMMODITY.isNotNull(),
                                          source_df.FINAL_REPORT_UNSPSC_COMMODITY[0:4]))
    source_df = source_df.withColumn("FINAL_REPORT_UNSPSC_CLASS",
                                     when(source_df.FINAL_REPORT_UNSPSC_COMMODITY.isNotNull(),
                                          source_df.FINAL_REPORT_UNSPSC_COMMODITY[0:6]))
    return source_df


