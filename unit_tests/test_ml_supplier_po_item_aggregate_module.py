from datetime import datetime, timedelta

import pytest
import mock
import sys

sys.path.append("../modules")
import ml_supplier_po_item_aggregate_module
@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_supplier_po_item_aggregate_module.sbnutils") as mock_sbnutils:
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
            "ml_supplier_po_item_aggregate_module.current_timestamp") as mock_current_timestamp:
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
def fix_when():
    with mock.patch("ml_supplier_po_item_aggregate_module.when") as mock_when:
        yield mock_when


@pytest.fixture
def fix_col():
    with mock.patch("ml_supplier_po_item_aggregate_module.col") as mock_col:
        yield mock_col


@pytest.fixture
def fix_isnull():
    with mock.patch("ml_supplier_po_item_aggregate_module.isnull") as mock_isnull:
        yield mock_isnull


@pytest.fixture
def fix_concat():
    with mock.patch("ml_supplier_po_item_aggregate_module.concat") as mock_concat:
        yield mock_concat


@pytest.fixture
def fix_year():
    with mock.patch("ml_supplier_po_item_aggregate_module.year") as mock_year:
        yield mock_year


@pytest.fixture
def fix_month():
    with mock.patch("ml_supplier_po_item_aggregate_module.month") as mock_month:
        yield mock_month


@pytest.fixture
def fix_lpad():
    with mock.patch("ml_supplier_po_item_aggregate_module.lpad") as mock_lpad:
        yield mock_lpad


@pytest.fixture
def fix_lit():
    with mock.patch("ml_supplier_po_item_aggregate_module.lit") as mock_lit:
        yield mock_lit


@pytest.fixture
def fix_sum():
    with mock.patch("ml_supplier_po_item_aggregate_module.sum") as mock_sum:
        yield mock_sum

@pytest.fixture
def fix_avg():
    with mock.patch("ml_supplier_po_item_aggregate_module.avg") as mock_avg:
        yield mock_avg


@pytest.fixture
def fix_countDistinct():
    with mock.patch("ml_supplier_po_item_aggregate_module.countDistinct") as mock_countDistinct:
        yield mock_countDistinct


def test_main(fix_sbnutils, fix_current_timestamp, fix_timestamp_range, fix_when, fix_col, fix_isnull, fix_concat, fix_month, fix_year, fix_lpad, fix_lit, fix_sum, fix_avg, fix_countDistinct):
    try:
        ml_supplier_po_item_aggregate_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_read_data(fix_sbnutils, fix_current_timestamp, fix_timestamp_range):
    _, mock_raw_data,mock_target_table = fix_sbnutils
    source_df = ml_supplier_po_item_aggregate_module._read_data(fix_source_table_location)
    assert source_df == mock_raw_data


def test_write_data(fix_sbnutils, fix_current_timestamp):
    _, mock_raw_data, mock_target_table = fix_sbnutils
    try:
        ml_supplier_po_item_aggregate_module._write_data(mock_raw_data, mock_target_table)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")