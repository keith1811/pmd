import pytest
import mock
import sys

sys.path.append("../modules")
import ml_replicate_data_from_raw_to_training_module

@pytest.fixture
def fix_sbnutils():
    with mock.patch("ml_replicate_data_from_raw_to_training_module.sbnutils") as mock_sbnutils:
        # Set up mock return values for sbnutils functions
        mock_sbnutils.get_table_storage_location.return_value = "test_source_table_location"
        mock_sbnutils.get_spark().read.format().load.return_value = mock.MagicMock()

        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdate().whenNotMatchedInsert().execute.return_value = (
            mock.MagicMock()
        )

        yield mock_sbnutils

def test_main(fix_sbnutils):
    try:
        ml_replicate_data_from_raw_to_training_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")