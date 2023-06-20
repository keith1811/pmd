from utils import sbnutils
from utils.constants import Zone, DELTA_UPDATED_FIELD, DELTA_CREATED_FIELD
from modules.utils.constants import MLTable
from pyspark.sql.types import StringType

# Copy commodity table from raw zone to consumption zone(manually, full replication);
def main():
    # Job Start
    sbnutils.log_info("Commodity table load job Start")

    source_zone = Zone.RAW.value
    source_table_name = MLTable.RAW_COMMODITY.value
    source_table_location = sbnutils.get_table_storage_location(source_zone, source_table_name)

    target_zone = Zone.CONSUMPTION.value
    target_table_name = MLTable.CONSUMPTION_COMMODITY.value
    target_table = sbnutils.get_delta_table(target_zone, target_table_name)

    # read all data of commodity table of raw zone
    commodity_df = _read_data(source_table_location)

    # cast code type from decimal to string
    commodity_df = _data_processing(commodity_df)

    sbnutils.log_info("Write to consumption commodity table")
    _write_data(commodity_df, target_table)

    sbnutils.log_info("Commodity table load job End")

def _read_data(source_table_location):
    spark = sbnutils.get_spark()

    source_df = (
        spark.read.format("delta").load(source_table_location)
    )
    return source_df

def _write_data(source_df, target_table):
    source_df = source_df.drop(DELTA_CREATED_FIELD, DELTA_UPDATED_FIELD)
    sbnutils.write_data(source_df, target_table, "ID")

def _data_processing(source_df):
    source_df.withColumn("CODE", source_df.CODE.cast(StringType()))
    return source_df
