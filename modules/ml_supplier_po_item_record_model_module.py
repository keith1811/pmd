from utils.constants import Zone
import utils.sbnutils as sbnutils
import uuid
import datetime
from modules.utils.config_utils import get_model_name, get_model_version, get_model_stage


def main():
    # Get env
    env = sbnutils._get_env()

    # Get model info
    model_name = get_model_name(env)
    model_stage = get_model_stage(env)
    model_version = get_model_version(env)

    # Create a dataframe of model information
    source_df = get_model_info(model_name, model_stage, model_version)

    # Target table
    target_table = sbnutils.get_delta_table(Zone.GENERAL.value, "ml_model_info")

    # Write model information into 'model_info'
    _write_data(source_df, target_table)


def get_model_info(model_name, model_stage, model_version):
    # Get spark and env ready
    spark = sbnutils.get_spark()

    # Prepare current_timestamp for _DELTA_CREATED_ON Timestamp
    timestamp_current = datetime.datetime.now()

    # Prepare uuid ( uuid.uuid4() ) for MODEL_UUID
    model_uuid = str(uuid.uuid4())

    # Create pyspark dataframe
    INFO_SCHEMA = """
              UUID String, NAME String, VERSION int, STAGE String, _DELTA_CREATED_ON Timestamp, _DELTA_UPDATED_ON Timestamp
    """
    df = spark.createDataFrame([
        (model_uuid, model_name, model_version, model_stage, timestamp_current, timestamp_current)
    ], schema=INFO_SCHEMA)
    return df


def _write_data(source_df, target_table):
    # Never update existed model & only record model information for unseen model
    (
        target_table.alias("target")
        .merge(source_df.alias("source"),
               "source.NAME = target.NAME AND source.VERSION = target.VERSION")
        .whenNotMatchedInsertAll()
        .execute()
    )
