from datetime import datetime, timedelta

import pytest
import mock
import sys

sys.path.append("../modules")
import ml_supplier_po_item_eval_descr_module


@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_supplier_po_item_eval_descr_module.sbnutils") as mock_sbnutils:
        mock_sbnutils.log_info().side_effect = None
        mock_sbnutils.get_table_storage_location().return_value = fix_source_table_location

        # mock spark read method
        mock_raw_data = mock.MagicMock()
        mock_raw_data.PROCESSED_DESCRIPTION = "asdfdadagadsfdagagadsdsga"
        mock_raw_data.DESCRIPTION = "asdfdadagadsfdagagadsdsga"
        mock_sbnutils.get_spark().read.format().load().where.return_value = (
            mock_raw_data
        )

        mock_commodity_data = mock.MagicMock()
        mock_sbnutils.get_spark().read.format().load().select.return_value = (
            mock_commodity_data
        )

        # mock spark write method
        mock_target_table = mock.MagicMock()
        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdateAll().whenNotMatchedInsertAll().execute.return_value = (
            mock_target_table
        )

        mock_clean_data = mock.MagicMock()
        mock_raw_data.withColumn().return_value = (
            mock_clean_data
        )

        mock_evaluate_description = mock.MagicMock()
        mock_clean_data.length()._lt_.return_value = False
        mock_clean_data.withColumn().return_value = (
            mock_evaluate_description
        )

        mock_evaluate_unspsc = mock.MagicMock()
        mock_evaluate_description.na.fill.withColumn().withColumn().join().withColumn().withColumn().withColumn().withColumn().withColumn().drop().return_value = (
            mock_evaluate_unspsc
        )

        mock_derive_unspsc = mock.MagicMock()
        mock_evaluate_unspsc.withColumn().withColumn().withColumn().withColumn().return_value = (
            mock_derive_unspsc
        )

        yield mock_sbnutils, mock_raw_data, mock_commodity_data, mock_target_table, mock_clean_data, mock_evaluate_description, mock_evaluate_unspsc, mock_derive_unspsc


@pytest.fixture
def fix_regexp_replace():
    with mock.patch(
            "ml_supplier_po_item_eval_descr_module.regexp_replace", return_value = "DESCRIPTION") as mock_regexp_replace:
        yield mock_regexp_replace


@pytest.fixture
def fix_lower():
    with mock.patch("ml_supplier_po_item_eval_descr_module.lower", return_value = "PROCESSED_DESCRIPTION") as mock_lower:
        yield mock_lower


@pytest.fixture
def fix_split():
    with mock.patch("ml_supplier_po_item_eval_descr_module.split", return_value = "PROCESSED_DESCRIPTION") as mock_split:
        yield mock_split


@pytest.fixture
def fix_ltrim():
    with mock.patch("ml_supplier_po_item_eval_descr_module.ltrim", return_value = "PROCESSED_DESCRIPTION") as mock_ltrim:
        yield mock_ltrim


@pytest.fixture
def fix_rtrim():
    with mock.patch("ml_supplier_po_item_eval_descr_module.rtrim", return_value = "PROCESSED_DESCRIPTION") as mock_rtrim:
        yield mock_rtrim


@pytest.fixture
def fix_array_distinct():
    with mock.patch("ml_supplier_po_item_eval_descr_module.array_distinct", return_value = "PROCESSED_DESCRIPTION") as mock_array_distinct:
        yield mock_array_distinct


@pytest.fixture
def fix_concat_ws():
    with mock.patch("ml_supplier_po_item_eval_descr_module.concat_ws", return_value = "PROCESSED_DESCRIPTION") as mock_concat_ws:
        yield mock_concat_ws


@pytest.fixture
def fix_length():
    with mock.patch("ml_supplier_po_item_eval_descr_module.length", return_value = 16) as mock_length:
        yield mock_length


@pytest.fixture
def fix_when():
    with mock.patch("ml_supplier_po_item_eval_descr_module.when") as mock_when:
        yield mock_when


@pytest.fixture
def fix_col():
    with mock.patch("ml_supplier_po_item_eval_descr_module.col") as mock_col:
        yield mock_col


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


def test_main(fix_sbnutils, fix_current_timestamp, fix_timestamp_range, fix_regexp_replace, fix_lower, fix_split,
              fix_array_distinct, fix_concat_ws, fix_ltrim, fix_rtrim, fix_length, fix_when, fix_col):
    try:
        ml_supplier_po_item_eval_descr_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_read_data(fix_sbnutils, fix_current_timestamp, fix_timestamp_range):
    _, mock_raw_data, mock_commodity_data, mock_target_table, mock_clean_data, mock_evaluate_description, mock_evaluate_unspsc, mock_derive_unspsc = fix_sbnutils
    source_df = ml_supplier_po_item_eval_descr_module._read_data(fix_source_table_location)
    assert source_df == mock_raw_data


def test_write_data(fix_sbnutils, fix_current_timestamp):
    _, mock_raw_data, mock_commodity_data, mock_target_table, mock_clean_data, mock_evaluate_description, mock_evaluate_unspsc, mock_derive_unspsc = fix_sbnutils
    try:
        ml_supplier_po_item_eval_descr_module._write_data(mock_raw_data, mock_target_table)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
