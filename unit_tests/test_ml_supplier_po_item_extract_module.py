import sys
import mock
import pytest

sys.path.append("../modules")
import ml_supplier_po_item_extract_module
from datetime import datetime, timedelta

@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_supplier_po_item_extract_module.sbnutils") as mock_sbnutils:
        mock_sbnutils.log_info().side_effect = None
        mock_sbnutils.get_table_storage_location().return_value = fix_source_table_location

        # mock spark read method
        mock_raw_data = mock.MagicMock()
        mock_sbnutils.get_spark().read.format().load().where().selectExpr.return_value = (
            mock_raw_data
        )

        # mock spark write method
        mock_target_table = mock.MagicMock()
        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdateAll().whenNotMatchedInsertAll().execute.return_value = (
            mock_target_table
        )

        yield mock_sbnutils, mock_raw_data, mock_target_table


@pytest.fixture
def fix_dataframe():
    mock_dataframe = mock.MagicMock()
    mock_dataframe.limit.return_value = mock_dataframe
    yield mock_dataframe


@pytest.fixture
def fix_lit():
    with mock.patch("ml_supplier_po_item_extract_module.lit") as mock_lit:
        yield mock_lit

@pytest.fixture
def fix_col():
    with mock.patch("ml_supplier_po_item_extract_module.col") as mock_col:
        yield mock_col

@pytest.fixture
def fix_concat():
    with mock.patch("ml_supplier_po_item_extract_module.concat") as mock_concat:
        yield mock_concat

@pytest.fixture
def fix_source_table_location():
    return "test_source_table_location"


@pytest.fixture
def fix_current_timestamp():
    with mock.patch("ml_supplier_po_item_extract_module.current_timestamp") as mock_current_timestamp:
        mock_current_timestamp.return_value = datetime.now()
        yield mock_current_timestamp


@pytest.fixture
def fix_timestamp_range():
    with mock.patch("utils.batch_utils.get_batch_timestamp_range") as mock_timestamp_range:
        current_time = datetime.now()
        yesterday_time = current_time - timedelta(1)
        mock_timestamp_range = [yesterday_time, current_time]
        yield mock_timestamp_range


def test_main(fix_sbnutils, fix_timestamp_range, fix_current_timestamp, fix_lit, fix_col, fix_concat):
    try:
        ml_supplier_po_item_extract_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_read_data(fix_sbnutils, fix_current_timestamp, fix_timestamp_range):
    _, mock_raw_data, mock_target_table = fix_sbnutils
    source_df = ml_supplier_po_item_extract_module._read_data(fix_source_table_location, [], "where_condition")
    assert source_df == mock_raw_data


def test_write_data(fix_sbnutils, fix_current_timestamp):
    _, mock_raw_data, mock_target_table = fix_sbnutils
    try:
        ml_supplier_po_item_extract_module._write_data(mock_raw_data, mock_target_table)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")



