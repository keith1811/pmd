import pytest
import mock
import sys
from modules.utils.config_utils import get_model_name, get_model_version, get_model_stage

sys.path.append("../modules")
import ml_supplier_po_item_record_model_module


@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_supplier_po_item_record_model_module.sbnutils") as mock_sbnutils:
        mock_sbnutils.log_info().side_effect = None
        mock_sbnutils.get_table_storage_location().return_value = "test_source_table_location"
        mock_sbnutils._get_env.return_value = "notebookdev"

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



@mock.patch("ml_supplier_po_item_record_model_module.get_model_info")
@mock.patch("ml_supplier_po_item_record_model_module._write_data")
def test_main(get_model_info, _write_data, fix_sbnutils):
    try:
        ml_supplier_po_item_record_model_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_get_model_info(spark):
    env = "notebookdev"
    model_name = get_model_name(env)
    model_stage = get_model_stage(env)
    model_version = get_model_version(env)
    df = ml_supplier_po_item_record_model_module.get_model_info(model_name, model_stage, model_version)

    assert type(model_name) == str
    assert model_name == get_model_name(env)
    assert type(model_stage) == str
    assert model_stage == get_model_stage(env)
    assert type(model_version) == int
    assert model_version == get_model_version(env)
    assert df.collect()[0]['NAME'] == model_name
    assert df.collect()[0]['VERSION'] == model_version
    assert df.collect()[0]['STAGE'] == model_stage


def test_write_data(fix_sbnutils):
    _, mock_raw_data, mock_target_table = fix_sbnutils
    try:
        ml_supplier_po_item_record_model_module._write_data(mock_raw_data, mock_target_table)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
