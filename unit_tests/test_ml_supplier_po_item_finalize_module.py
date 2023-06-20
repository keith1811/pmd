import pytest
import mock
import sys
from datetime import datetime, timedelta
from pyspark.sql import SparkSession

sys.path.append("../modules")
import ml_supplier_po_item_finalize_module

@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_supplier_po_item_finalize_module.sbnutils") as mock_sbnutils:
        mock_sbnutils.log_info().side_effect = None
        mock_sbnutils.get_table_storage_location().return_value = fix_source_table_location

        # mock spark read method
        mock_raw_data = mock.MagicMock()
        mock_sbnutils.get_spark().read.format().load().where.return_value = (
            mock_raw_data
        )

        # mock spark write method
        mock_target_table = mock.MagicMock()
        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdateAll().whenNotMatchedInsertAll().execute.return_value = (
            mock_target_table
        )
        yield mock_sbnutils, mock_raw_data, mock_target_table


@pytest.fixture
def fix_source_table_location():
    return "test_source_table_location"


@pytest.fixture
def fix_current_timestamp():
    with mock.patch(
            "ml_supplier_po_item_eval_descr_module.current_timestamp") as mock_current_timestamp:
        mock_current_timestamp.return_value = datetime.now()
        yield mock_current_timestamp


@pytest.fixture
def fix_timestamp_range():
    with mock.patch("utils.batch_utils.get_batch_timestamp_range") as mock_timestamp_range:
        current_time = datetime.now()
        yesterday_time = current_time - timedelta(1)
        mock_timestamp_range = [yesterday_time, current_time]
        yield mock_timestamp_range


@pytest.fixture
def fix_source_table(spark):
    columns = ["ID", "PO_ITEM_ID", "AN_UNSPSC_COMMODITY", "EXTERNAL_PREDICATED_UNSPSC_COMMODITY",
               "EXTERNAL_PREDICATION_CONFIDENCE_COMMODITY", "SBN_PREDICATED_UNSPSC_COMMODITY",
               "SBN_PREDICTION_CONFIDENCE_COMMODITY", "FINAL_REPORT_UNSPSC_COMMODITY", "FINAL_REPORT_CONFIDENCE_COMMODITY",
               "_DELTA_CREATED_ON", "_DELTA_UPDATED_ON", "AN_CLASSIFICATION_QUALITY_COMMODITY"]
    data = [
        ("1001", 10001, '11203012', '11203010', 0.9, '11203012', 0.65, '', '', '', '', ''),
        ("1002", 10002, '11203012', '11203010', 0.6, '11203013', 0.89, '', '', '', '', ''),
        ("1003", 10003, '11203012', '11203012', 0.7, '11203013', 0.6, '', '', '', '', ''),
        ("1004", 10004, '11203012', '11203013', 0.7, '11203013', 0.6, '', '', '', '', ''),
        ("1005", 10005, '11203012', '11203013', 0.7, '11203013', 0.6, '', '', '', '', 'Good')
    ]
    df = spark.createDataFrame(data).toDF(*columns)
    return df

@mock.patch("ml_supplier_po_item_finalize_module._read_data")
@mock.patch("ml_supplier_po_item_finalize_module._write_data")
@mock.patch("ml_supplier_po_item_finalize_module._finalize_unspsc_commodity")
@mock.patch("ml_supplier_po_item_finalize_module._derive_unspsc")
def test_main(mock_read_data, mock_write_data, mock_finalize_unspsc_commodity, mock_derive_unspsc, fix_sbnutils):
    try:
        ml_supplier_po_item_finalize_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_read_data(fix_sbnutils, fix_current_timestamp, fix_timestamp_range):
    _, mock_raw_data, mock_target_table = fix_sbnutils
    source_df = ml_supplier_po_item_finalize_module._read_data(fix_source_table_location)
    assert source_df == mock_raw_data


def test_write_data(fix_sbnutils, spark):
    _, mock_raw_data, mock_target_table = fix_sbnutils
    try:
        ml_supplier_po_item_finalize_module._write_data(mock_raw_data, mock_target_table)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_finalize_unspsc_commodity(fix_source_table, spark, fix_sbnutils):
    assert fix_source_table.collect()[0]["ID"] == '1001'
    assert fix_source_table.count() == 5
    source_table = ml_supplier_po_item_finalize_module._finalize_unspsc_commodity(fix_source_table)
    assert source_table.filter(source_table.ID == "1001").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203010'
    assert source_table.filter(source_table.ID == "1001").collect()[0]["FINAL_REPORT_CONFIDENCE_COMMODITY"] == 0.9
    assert source_table.filter(source_table.ID == "1002").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203013'
    assert source_table.filter(source_table.ID == "1002").collect()[0]["FINAL_REPORT_CONFIDENCE_COMMODITY"] == 0.89
    assert source_table.filter(source_table.ID == "1003").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203012'
    assert source_table.filter(source_table.ID == "1003").collect()[0]["FINAL_REPORT_CONFIDENCE_COMMODITY"] == 0.7
    assert source_table.filter(source_table.ID == "1004").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] is None
    assert source_table.filter(source_table.ID == "1004").collect()[0]["FINAL_REPORT_CONFIDENCE_COMMODITY"] is None
    assert source_table.filter(source_table.ID == "1005").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203012'
    assert source_table.filter(source_table.ID == "1005").collect()[0]["FINAL_REPORT_CONFIDENCE_COMMODITY"] is None


def test_derive_unspsc(fix_source_table, spark, fix_sbnutils):
    source_table = ml_supplier_po_item_finalize_module._derive_unspsc(ml_supplier_po_item_finalize_module._finalize_unspsc_commodity(fix_source_table))
    assert source_table.count() == 5
    assert source_table.filter(source_table.ID == "1001").collect()[0]["FINAL_REPORT_UNSPSC_SEGMENT"] == '11'
    assert source_table.filter(source_table.ID == "1001").collect()[0]["FINAL_REPORT_UNSPSC_FAMILY"] == '1120'
    assert source_table.filter(source_table.ID == "1001").collect()[0]["FINAL_REPORT_UNSPSC_CLASS"] == '112030'
    assert source_table.filter(source_table.ID == "1001").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203010'
    assert source_table.filter(source_table.ID == "1002").collect()[0]["FINAL_REPORT_UNSPSC_SEGMENT"] == '11'
    assert source_table.filter(source_table.ID == "1002").collect()[0]["FINAL_REPORT_UNSPSC_FAMILY"] == '1120'
    assert source_table.filter(source_table.ID == "1002").collect()[0]["FINAL_REPORT_UNSPSC_CLASS"] == '112030'
    assert source_table.filter(source_table.ID == "1002").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203013'
    assert source_table.filter(source_table.ID == "1003").collect()[0]["FINAL_REPORT_UNSPSC_SEGMENT"] == '11'
    assert source_table.filter(source_table.ID == "1003").collect()[0]["FINAL_REPORT_UNSPSC_FAMILY"] == '1120'
    assert source_table.filter(source_table.ID == "1003").collect()[0]["FINAL_REPORT_UNSPSC_CLASS"] == '112030'
    assert source_table.filter(source_table.ID == "1003").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203012'
    assert source_table.filter(source_table.ID == "1004").collect()[0]["FINAL_REPORT_UNSPSC_SEGMENT"] is None
    assert source_table.filter(source_table.ID == "1004").collect()[0]["FINAL_REPORT_UNSPSC_FAMILY"] is None
    assert source_table.filter(source_table.ID == "1004").collect()[0]["FINAL_REPORT_UNSPSC_CLASS"] is None
    assert source_table.filter(source_table.ID == "1004").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] is None
    assert source_table.filter(source_table.ID == "1005").collect()[0]["FINAL_REPORT_UNSPSC_SEGMENT"] == '11'
    assert source_table.filter(source_table.ID == "1005").collect()[0]["FINAL_REPORT_UNSPSC_FAMILY"] == '1120'
    assert source_table.filter(source_table.ID == "1005").collect()[0]["FINAL_REPORT_UNSPSC_CLASS"] == '112030'
    assert source_table.filter(source_table.ID == "1005").collect()[0]["FINAL_REPORT_UNSPSC_COMMODITY"] == '11203012'
