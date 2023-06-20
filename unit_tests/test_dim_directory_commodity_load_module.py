import pytest
import mock
import sys

sys.path.append("../modules")
import dim_directory_commodity_load_module

@pytest.fixture
def fix_sbnutils():
    with mock.patch("dim_directory_commodity_load_module.sbnutils") as mock_sbnutils:
        # Set up mock return values for sbnutils functions
        mock_sbnutils.get_table_storage_location.return_value = fix_source_table_location

        # mock spark.readStream...load() method
        mock_sbnutils.get_spark().read.format().load.return_value = mock.MagicMock()

        mock_sbnutils.get_delta_table().alias().merge().whenMatchedUpdate().whenNotMatchedInsert().execute.return_value = (
            mock.MagicMock()
        )

        yield mock_sbnutils

@pytest.fixture
def fix_source_table_location():
    return "test_source_table_location"

def test_main(
    fix_source_table_location,
    fix_sbnutils
):
    try:
        dim_directory_commodity_load_module.main()
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")