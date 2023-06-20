import mock
import pytest
import utils.constants as sbn_constants
from modules.utils.persist_with_dbutils import *
from modules.utils.common import MLflowTrackingConfig
from pyspark.sql.types import *
from decimal import Decimal

@pytest.fixture
def sample_dataframe_1(spark):
    """Mock model info df."""
    schema = StructType(
        [
            StructField("UUID", StringType(), True),
            StructField("NAME", StringType(), True),
            StructField("VERSION", IntegerType(), True),
            StructField("STAGE", StringType(), True),
            StructField("CONCAT_FEATURE", StringType(), True),
            StructField("UNSPSC_SEGMENT", StringType(), True),
            StructField("UNSPSC_FAMILY", StringType(), True),
            StructField("UNSPSC_CLASS", StringType(), True),
            StructField("UNSPSC_COMMODITY", StringType(), True),
            StructField("CONFIDENCE_SEGMENT", DecimalType(6, 5), True),
            StructField("CONFIDENCE_FAMILY", DecimalType(6, 5), True),
            StructField("CONFIDENCE_CLASS", DecimalType(6, 5), True),
            StructField("CONFIDENCE_COMMODITY", DecimalType(6, 5), True),
            StructField("REPORT_LASTUPDATED_AT", TimestampType(), True),
            StructField("MODEL_UUID", StringType(), True),
            StructField(f"{sbn_constants.DELTA_CREATED_FIELD}", TimestampType(), True),
            StructField(f"{sbn_constants.DELTA_UPDATED_FIELD}", TimestampType(), True)
        ]
    )

    data = [("123", "test", 1, "Staging", "test description", "10", "1010", "101010", "10101010", 
             *(Decimal(0.78900) for _ in range(4)), None, 'MID-001', None, None)]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def fix_source_df(spark):
    schema = StructType(
        [
            StructField("ID", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("CONCAT_FEATURE", StringType(), True),
            StructField("predicted_label", StringType(), True),
            StructField("predicted_label_proba", DecimalType(6,5), True),
            StructField("SBN_PREDICTION_LASTUPDATED_AT", TimestampType(), True),
            StructField("REPORT_LASTUPDATED_AT", TimestampType(), True),
            StructField("MODEL_UUID", StringType(), True),
            StructField(f"{sbn_constants.DELTA_CREATED_FIELD}", TimestampType(), True),
            StructField(f"{sbn_constants.DELTA_UPDATED_FIELD}", TimestampType(), True)
        ]
    )
    data = [
        ("1", "10001",'test description manufacture name N/A',"10101010", Decimal(0.78900), None, None, 'MID-001', None, None)
    ]
    return spark.createDataFrame(data, schema)

@pytest.fixture
def fix_source_table_location():
    return "test_source_table_location"

@pytest.fixture(scope="function")
def fix_sbnutils(sample_dataframe_1):
    with mock.patch("modules.utils.persist_with_dbutils.sbnutils") as mock_sbnutils:
        # Set up mock return values for sbnutils functions
        mock_sbnutils.get_table_storage_location.return_value = fix_source_table_location
        mock_sbnutils.get_spark().read.format().load().where.return_value = sample_dataframe_1
        yield mock_sbnutils, sample_dataframe_1

def test_persist_to_ml_supplier_po_item(fix_source_df, fix_sbnutils):
    mlflow_tracking_cfg = MLflowTrackingConfig(model_name="po_classification",
                                            model_registry_stage="Staging",
                                            model_version=1)
    persist_to_ml_supplier_po_item(fix_source_df, mlflow_tracking_cfg)

def test_persist_to_bna_report(fix_source_df, fix_sbnutils):
    mlflow_tracking_cfg = MLflowTrackingConfig(model_name="po_classification",
                                            model_registry_stage="Staging",
                                            model_version=1)
    persist_to_bna_report(fix_source_df, mlflow_tracking_cfg)


def test_split_inference_dataframe(fix_source_df, fix_sbnutils):
    mlflow_tracking_cfg = MLflowTrackingConfig(model_name="po_classification",
                                            model_registry_stage="Staging",
                                            model_version=1)
    split_inference_dataframe(fix_source_df, mlflow_tracking_cfg)


def test_get_highest_predict_level():
    code_1 = "23101010"
    code_2 = "23000000"
    code_3 = "23100000"
    level_1 = get_highest_predict_level(code_1)
    level_2 = get_highest_predict_level(code_2)
    level_3 = get_highest_predict_level(code_3)
    assert level_1 == 4
    assert level_2 == 1
    assert level_3 == 2

def test_get_level_confidence():
    actual_confidence_1 = get_level_confidence(code="23100000",prob=0.78910,level=1)
    actual_confidence_2 = get_level_confidence(code="23230000",prob=0.78910,level=3)
    actual_confidence_3 = get_level_confidence(code="23100000",prob=0.78910,level=2)
    actual_confidence_4 = get_level_confidence(code="23100000",prob=0.78910,level=4)
    assert 0.78910 == actual_confidence_1
    assert actual_confidence_2 == None
    assert 0.78910 == actual_confidence_3
    assert actual_confidence_4 == None

def test_get_level_code():
    # case 1: 10100000
    actual_code_1_1 = get_level_code(code="10100000",level=1)
    assert "10" == actual_code_1_1
    actual_code_1_2 =  get_level_code(code="10100000",level=2)
    assert "1010" == actual_code_1_2
    actual_code_1_3 =  get_level_code(code="10100000",level=3)
    assert actual_code_1_3 == None
    actual_code_1_4 =  get_level_code(code="10100000",level=4)
    assert actual_code_1_4 == None
    # case 2: 11111111
    actual_code_2_1 = get_level_code(code="11111111",level=1)
    assert "11" == actual_code_2_1
    actual_code_2_2 =  get_level_code(code="11111111",level=2)
    assert "1111" == actual_code_2_2
    actual_code_2_3 =  get_level_code(code="11111111",level=3)
    assert "111111" == actual_code_2_3
    actual_code_2_4 =  get_level_code(code="11111111",level=4)
    assert "11111111" == actual_code_2_4
