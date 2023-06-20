import sys
import mock
import pytest

sys.path.append("../modules")
import ml_supplier_po_item_load_module
from datetime import datetime

@pytest.fixture
def fix_ml_supplier_po_item_table_columns():
    columns = [
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
        "REPORT_LASTUPDATED_AT",
        "REPORT_SOURCE"]
    return columns


@pytest.fixture
def fix_fact_supplier_po_item_columns():
    columns = ["ID as PO_ITEM_ID"]
    return columns


@pytest.fixture
def fix_table_storage_location():
    return "test_table_location"


@pytest.fixture
def fix_time_range():
    return ["2020-01-01 00:00:00", "2023-01-01 00:00:00"]


@pytest.fixture
def fix_delta_table():
    return "delta_table"


@pytest.fixture
def fix_batch_utils(fix_time_range):
    with mock.patch("ml_supplier_po_item_load_module.batch_utils") as mock_batch_utils:
        mock_batch_utils.get_batch_timestamp_range.return_value = fix_time_range
        yield mock_batch_utils


@pytest.fixture
def fix_current_timestamp():
    with mock.patch("ml_supplier_po_item_load_module.current_timestamp") as mock_timestamp:
        mock_timestamp.return_value = datetime.now()
        yield mock_timestamp


@pytest.fixture
def fix_datetime_strptime():
    with mock.patch("ml_supplier_po_item_load_module.datetime") as mock_strptime:
        yield mock_strptime


@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_supplier_po_item_load_module.sbnutils") as mock_sbnutils:
        mock_sbnutils.log_info().side_effect = None
        mock_sbnutils.get_table_storage_location.return_value = fix_table_storage_location

        # mock dbutils
        mock_dbutils = mock.MagicMock()
        mock_sbnutils.get_dbutils.return_value = mock_dbutils

        # mock spark read method
        mock_ml_supplier_po_item_df = mock.MagicMock()
        mock_sbnutils.get_spark().read.format().load().where().selectExpr.return_value = (
            mock_ml_supplier_po_item_df
        )

        # mock spark write to dim table method
        mock_dim_table = mock.MagicMock()
        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdate().whenNotMatchedInsert().execute.return_value = (
            mock_dim_table
        )

        # mock spark write to fact table
        mock_fact_table = mock.MagicMock()
        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdate().execute.return_value = (
            mock_fact_table
        )

        yield mock_sbnutils, mock_ml_supplier_po_item_df, mock_dim_table, mock_fact_table


def test_read_data(fix_sbnutils, fix_ml_supplier_po_item_table_columns):
    _, mock_ml_supplier_po_item_df, _, _ = fix_sbnutils
    source_df = ml_supplier_po_item_load_module._read_data(fix_table_storage_location, fix_ml_supplier_po_item_table_columns)
    assert source_df == mock_ml_supplier_po_item_df


def test_write_data_to_dim(fix_sbnutils, fix_current_timestamp):
    _,  mock_source_df, mock_dim_table, _ = fix_sbnutils
    ml_supplier_po_item_load_module._write_data_to_dim(mock_source_df, mock_dim_table)


def test_write_data_to_fact(fix_sbnutils, fix_current_timestamp):
    _,  mock_source_df, _, mock_fact_table = fix_sbnutils
    ml_supplier_po_item_load_module._write_data_to_fact(mock_source_df, mock_fact_table)
    

def test_main(
    fix_sbnutils,
    fix_batch_utils,
    fix_datetime_strptime,
    fix_current_timestamp
):
    try:
        ml_supplier_po_item_load_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")