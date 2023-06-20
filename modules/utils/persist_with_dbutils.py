import pyspark
import utils.sbnutils as sbnutils
import utils.constants as sbn_constants
import math
from pyspark.sql.functions import current_timestamp, col, lit, udf
from modules.utils.constants import MLTable
from modules.utils.common import MLflowTrackingConfig

ml_po_item_update_columns = [
    "ID",
    "SBN_PREDICATED_UNSPSC_SEGMENT",
    "SBN_PREDICATED_UNSPSC_FAMILY",
    "SBN_PREDICATED_UNSPSC_CLASS",
    "SBN_PREDICATED_UNSPSC_COMMODITY",
    "SBN_PREDICTION_CONFIDENCE_SEGMENT",
    "SBN_PREDICTION_CONFIDENCE_FAMILY",
    "SBN_PREDICTION_CONFIDENCE_CLASS",
    "SBN_PREDICTION_CONFIDENCE_COMMODITY",
    "MODEL_UUID",
    "SBN_PREDICTION_LASTUPDATED_AT",
    "_DELTA_UPDATED_ON"
]

def get_highest_predict_level(code: str)->int:
    clean_code = code.rstrip("0")
    highest_level = math.ceil(len(clean_code)/2)
    return highest_level

def get_level_confidence(code, prob, level):
    highest_level = get_highest_predict_level(code)
    if level > highest_level: return None
    return prob
confidenceUDF = udf(lambda a,b,c: get_level_confidence(a,b,c))

def get_level_code(code, level):
    code_1 = code[0:2]
    code_2 = code[0:4]
    code_3 = code[0:6]
    code_4 = code[0:8]

    codes = [code_1, code_2, code_3, code_4]

    num_of_r_zero = 10-2*level
    if "0"*num_of_r_zero == code[-num_of_r_zero:]:
        codes[level-1] = None
    return codes[level-1]
codeUDF = udf(lambda a,b: get_level_code(a,b))

def generate_code_and_confidence_columns(df, prefix):
    sbnutils.log_info("Generate predict code and confidence in four levels")
    code_prefix = ""
    prob_prefix = ""
    if prefix in ["sbn", "SBN"]:
        code_prefix = "SBN_PREDICATED_"
        prob_prefix = "SBN_PREDICTION_"

    df = (
        df.withColumn(f"{code_prefix}UNSPSC_SEGMENT", codeUDF(df['predicted_label'],lit(1)))
        .withColumn(f"{code_prefix}UNSPSC_FAMILY", codeUDF(df['predicted_label'],lit(2)))
        .withColumn(f"{code_prefix}UNSPSC_CLASS", codeUDF(df['predicted_label'],lit(3)))
        .withColumn(f"{code_prefix}UNSPSC_COMMODITY", codeUDF(df['predicted_label'],lit(4)))
        .withColumn(f"{prob_prefix}CONFIDENCE_SEGMENT", confidenceUDF(df['predicted_label'], df['predicted_label_proba'], lit(1)))
        .withColumn(f"{prob_prefix}CONFIDENCE_FAMILY", confidenceUDF(df['predicted_label'], df['predicted_label_proba'], lit(2)))
        .withColumn(f"{prob_prefix}CONFIDENCE_CLASS", confidenceUDF(df['predicted_label'], df['predicted_label_proba'], lit(3)))
        .withColumn(f"{prob_prefix}CONFIDENCE_COMMODITY", confidenceUDF(df['predicted_label'], df['predicted_label_proba'], lit(4)))
    )
    
    return df

def persist_to_bna_report(df: pyspark.sql.DataFrame, cfg: MLflowTrackingConfig):
    model_info_df = search_model_info_table_by_model_name_and_version(model_name=cfg.model_name,
                                                                      model_version=cfg.model_version)
    model_uuid = model_info_df.collect()[0]["UUID"]

    cur_timestamp = current_timestamp()

    df = generate_code_and_confidence_columns(df, prefix="")

    df = (
        df.withColumn("MODEL_UUID", lit(model_uuid))
        .withColumn("REPORT_LASTUPDATED_AT", cur_timestamp)
        .withColumn("_DELTA_CREATED_ON", cur_timestamp)
        .withColumn("_DELTA_UPDATED_ON", cur_timestamp)
        .dropDuplicates(["CONCAT_FEATURE"])
    )

    _write_to_bna_report_table(df)


def persist_to_ml_supplier_po_item(df: pyspark.sql.DataFrame, cfg: MLflowTrackingConfig):
    model_info_df = search_model_info_table_by_model_name_and_version(model_name=cfg.model_name,
                                                                      model_version=cfg.model_version)
    model_uuid = model_info_df.collect()[0]["UUID"]

    cur_timestamp = current_timestamp()

    df = generate_code_and_confidence_columns(df, prefix="sbn")

    df = (
        df.withColumn("PO_ITEM_ID", col("PO_ITEM_ID").cast("Decimal(28)"))
        .withColumn("MODEL_UUID", lit(model_uuid))
        .withColumn("SBN_PREDICTION_LASTUPDATED_AT", cur_timestamp)
        .withColumn("_DELTA_UPDATED_ON", cur_timestamp)
        .selectExpr(ml_po_item_update_columns)
    )

    _write_to_po_item_table(df)

def search_model_info_table_by_model_name_and_version(model_name: str = None,
                                             model_version: int = None) -> pyspark.sql.DataFrame:
    spark = sbnutils.get_spark()

    model_info_table_location = sbnutils.get_table_storage_location(sbn_constants.Zone.GENERAL.value,
                                                                    MLTable.GENERAL_ML_MODEL_INFO.value)

    if model_name and model_version:
        source_df = (
            spark.read.format("delta")
                .load(model_info_table_location)
                .where(f"""
                NAME == '{model_name}' AND VERSION == {model_version}
            """)
        )
        return source_df


def check_model_stage_consistence(cfg: MLflowTrackingConfig):
    # get model info from model info table
    model_info_df = search_model_info_table_by_model_name_and_version(model_name=cfg.model_name,
                                                                      model_version=cfg.model_version)

    if model_info_df.count() == 0: # no model info in table
        raise RuntimeError(f"No model named {cfg.model_name}, version {cfg.model_version} in model info table")
    else:
        model_stage_info = model_info_df.collect()[0]["STAGE"]
        if cfg.model_registry_stage != model_stage_info:
            raise RuntimeError(f"Model stage inconsistent, config model stage: {cfg.model_registry_stage}, model info stage: {model_stage_info}")


def split_inference_dataframe(df: pyspark.sql.DataFrame, cfg: MLflowTrackingConfig):
    spark = sbnutils.get_spark()
    model_info_df = search_model_info_table_by_model_name_and_version(model_name=cfg.model_name,
                                                                      model_version=cfg.model_version)
    model_uuid = model_info_df.collect()[0]["UUID"]

    bna_report_table_location = sbnutils.get_table_storage_location(sbn_constants.Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_BNA_UPSPSC_REPORT.value)

    report_df = (
        spark.read.format("delta")
        .load(bna_report_table_location)
        .where(f""" MODEL_UUID = '{model_uuid}' """)
        .withColumnRenamed("UNSPSC_SEGMENT", "SBN_PREDICATED_UNSPSC_SEGMENT")
        .withColumnRenamed("UNSPSC_FAMILY", "SBN_PREDICATED_UNSPSC_FAMILY")
        .withColumnRenamed("UNSPSC_CLASS", "SBN_PREDICATED_UNSPSC_CLASS")
        .withColumnRenamed("UNSPSC_COMMODITY", "SBN_PREDICATED_UNSPSC_COMMODITY")
        .withColumnRenamed("CONFIDENCE_SEGMENT", "SBN_PREDICTION_CONFIDENCE_SEGMENT")
        .withColumnRenamed("CONFIDENCE_FAMILY", "SBN_PREDICTION_CONFIDENCE_FAMILY")
        .withColumnRenamed("CONFIDENCE_CLASS", "SBN_PREDICTION_CONFIDENCE_CLASS")
        .withColumnRenamed("CONFIDENCE_COMMODITY", "SBN_PREDICTION_CONFIDENCE_COMMODITY")
        .withColumnRenamed("REPORT_LASTUPDATED_AT", "SBN_PREDICTION_LASTUPDATED_AT")
    )

    cur_timestamp = current_timestamp()
    no_need_inference_df = (
        df.select(["ID","PO_ITEM_ID","CONCAT_FEATURE"])
        .join(report_df, on='CONCAT_FEATURE', how='inner')
        .withColumn("_DELTA_UPDATED_ON", cur_timestamp)
        .selectExpr(ml_po_item_update_columns)
    )

    need_inference_df = df.join(report_df, on='CONCAT_FEATURE', how='left_anti').selectExpr(df.columns)

    sbnutils.log_info(f"Write no need inferenced data into po_item table")
    _write_to_po_item_table(no_need_inference_df)

    return need_inference_df


def _write_to_po_item_table(df):
    target_table = sbnutils.get_delta_table(sbn_constants.Zone.ENRICHMENT.value, sbn_constants.Table.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value)
    target_columns = df.columns
    update_expr = {f"target.{c}": f"source.{c}" for c in target_columns}

    merge_condition = f"""
        source.ID = target.ID and (
        target.MODEL_UUID is NULL
        or source.MODEL_UUID != target.MODEL_UUID)
    """
    (
        target_table.alias("target")
            .merge(df.alias("source"), merge_condition)
            .whenMatchedUpdate(set=update_expr)
            .execute()
    )

def _write_to_bna_report_table(df):
    target_table = sbnutils.get_delta_table(sbn_constants.Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_BNA_UPSPSC_REPORT.value)
    target_columns = target_table.toDF().columns

    update_expr = {f"target.{c}": f"source.{c}" for c in target_columns}
    insert_expr = update_expr.copy()

    update_expr.pop(f"target.{sbn_constants.DELTA_CREATED_FIELD}", None)
    update_expr.pop(f"{sbn_constants.DELTA_CREATED_FIELD}", None)

    (
        target_table.alias("target")
            .merge(df.alias("source"), "source.CONCAT_FEATURE = target.CONCAT_FEATURE and source.MODEL_UUID = target.MODEL_UUID")
            .whenMatchedUpdate(set=update_expr)
            .whenNotMatchedInsert(values=insert_expr)
            .execute()
    )