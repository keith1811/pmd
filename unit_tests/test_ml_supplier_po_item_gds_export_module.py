import pytest
from datetime import datetime,timedelta
import os
import mock
import sys
from pyspark.sql.types import StringType, LongType, TimestampType, StructType, StructField, DecimalType
from decimal import Decimal
from modules.utils.constants import MLTable
from delta.tables import DeltaTable
from utils.constants import Zone
from utils import sbnutils

sys.path.append("../modules")
import ml_supplier_po_item_gds_export_module

timestamp_earliest = datetime.strptime("2020-01-01 00:00:00.000", "%Y-%m-%d %H:%M:%S.%f")
timestamp_before_earliest = timestamp_earliest - timedelta(microseconds=1)
timestamp_current = datetime.now()
timestamp_before_current = timestamp_current - timedelta(minutes=60)
timestamp_after_current = timestamp_current + timedelta(minutes=60)

@pytest.fixture(scope="module")
def create_supplier_po_item_df(spark, current_test_dir, base_testdata_path):
    """Mock ml_supplier_po_item source df."""
    schema = StructType(
        [
            StructField("ID", StringType(), True),
            StructField("PO_ITEM_ID", LongType(), True),
            StructField("DASHBOARD_STATUS", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("PROCESSED_DESCRIPTION", StringType(), True),
            StructField("AN_DATA_QUALITY_LEVEL", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("MANUFACTURER_NAME", StringType(), True),
            StructField("AN_UNSPSC_COMMODITY", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_SEGMENT", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_FAMILY", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_CLASS", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_COMMODITY", StringType(), True),
            StructField("_DELTA_CREATED_ON", TimestampType(), True),
            StructField("_DELTA_UPDATED_ON", TimestampType(), True)
        ]
    )

    df = spark.createDataFrame([
            ("1", 1001, 'PO_NEW', 'aa', 'aa', 'Good', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_before_earliest),
            ("2", 1002, 'PO_OBSOLETED', 'BB', 'bb', 'Good', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_after_current),
            ("3", 1004, 'PO_PARTIALLY_SERVICED', 'cc', 'cc', 'Poor', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_current),
            ("4", 1003, 'PO_NEW', 'cc4', 'cc', 'Poor', 'mp', 'mn', '12345678',None, None, None, None, timestamp_current, timestamp_current),
            ("5", 1005, 'PO_SHIPPED','dd', 'dd', 'Good', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_current),
            ("6", 1006, 'PO_NEW', 'dd4,|', 'dd', 'Good', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_current),
            ("7", 1007, 'PO_NEW', 'ee', 'ee', 'Good', 'mp', 'mn', '12345678', '12', None, None, None, timestamp_current, timestamp_current),
            ("8", 1008, 'PO_NEW', 'ff', 'ff', 'Good', 'mp', 'mn', '12345678', '', None, None, None, timestamp_current, timestamp_current),
            ("9", 1009, 'PO_NEW', 'g///@g', 'gg', 'Good', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_current),
            ("10", 1010, 'PO_OBSOLETED', 'hh', 'hh', 'Good', 'mp', 'mn', '12345678', None, None, None, None, timestamp_current, timestamp_current),
            ("11", 1010, 'PO_NEW', 'kk', 'kk', 'Good', 'mp', 'mn', '12345678', None, None, None, '', timestamp_current, timestamp_current),
    ], schema)
    df.coalesce(1).write.format("delta").mode('overwrite').option("header", "true").save(os.path.normpath(current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df"))
    return df

@pytest.fixture(scope="module")
def create_gds_unspsc_report_df(spark, current_test_dir, base_testdata_path):
    """Mock gds_unspsc_report df."""
    schema = StructType(
        [
            StructField("DESCRIPTION", StringType(), True),
            StructField("PROCESSED_DESCRIPTION", StringType(), True),
            StructField("UNSPSC_SEGMENT", StringType(), True),
            StructField("UNSPSC_FAMILY", StringType(), True),
            StructField("UNSPSC_CLASS", StringType(), True),
            StructField("UNSPSC_COMMODITY", StringType(), True),
            StructField("CONFIDENCE_SEGMENT", DecimalType(6,5), True),
            StructField("CONFIDENCE_FAMILY", DecimalType(6,5), True),
            StructField("CONFIDENCE_CLASS", DecimalType(6,5), True),
            StructField("CONFIDENCE_COMMODITY", DecimalType(6,5), True),
            StructField("_DELTA_CREATED_ON", TimestampType(), True),
            StructField("_DELTA_UPDATED_ON", TimestampType(), True)
        ]
    )

    spark.createDataFrame([
            ('d/d', 'dd', '12', '1234', '123456', '12345678', *(Decimal(0.78900) for _ in range(4)), timestamp_current, timestamp_current),
            ('e@e', 'ee', None, None, None, None, None, None, None, None, timestamp_current, timestamp_current),
            ('ff ', 'ff', None, None, None, None, None, None, None, None, timestamp_current, timestamp_current)
    ], schema).coalesce(1).write.format("delta").mode('overwrite').option("header", "true") \
    .save(os.path.normpath(current_test_dir + f"/{base_testdata_path}/gds_unspsc_report_df"))

@pytest.fixture
def get_table_storage_location_mocker(current_test_dir, base_testdata_path):
    with mock.patch("utils.sbnutils.get_table_storage_location") as table_storage_location:
        values = {(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value): os.path.normpath(current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df"),
                  (Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value): os.path.normpath(current_test_dir + f"/{base_testdata_path}/gds_unspsc_report_df")}
        table_storage_location.side_effect = mock.Mock(side_effect=lambda x, y: values[(x, y)])
        yield table_storage_location

@pytest.fixture
def get_location_base_mocker(current_test_dir, base_testdata_path):
    with mock.patch("utils.sbnutils._get_location_base") as location_base:
        location_base.return_value = os.path.normpath(
            current_test_dir + f"/{base_testdata_path}/sap_export_"
        )
        yield location_base

@pytest.fixture
def get_delta_table_mocker(spark, current_test_dir, base_testdata_path):
    with mock.patch("utils.sbnutils.get_delta_table") as delta_table:
        delta_table.return_value = DeltaTable.forPath(
            spark, current_test_dir + f"/{base_testdata_path}/gds_unspsc_report_df")
        yield delta_table

@pytest.fixture
def get_batch_timestamp_range_mocker(fix_time_range):
    with mock.patch("utils.batch_utils.get_batch_timestamp_range") as batch_timestamp_range:
        batch_timestamp_range.return_value = fix_time_range
        yield batch_timestamp_range

@pytest.fixture
def fix_time_range():
    return [timestamp_earliest, timestamp_current]

@pytest.fixture
def assertion():
    def assert_archive_file(file_name, number_of_archive_files, base_dir):
        archive_dir = f"sap_export_gds_archive"
        assert archive_dir in list(map(lambda fileInfo:fileInfo.name, sbnutils.get_dbutils().fs.ls(base_dir)))
        filenames = list(map(lambda fileInfo:fileInfo.name, sbnutils.get_dbutils().fs.ls(f"{base_dir}/{archive_dir}")))
        filenames.sort()
        assert len(filenames) == number_of_archive_files
        assert filenames[0] == f'{file_name}_{datetime.now().strftime("%Y_%m_%d")}'
        for i in range(1, number_of_archive_files):
            assert filenames[i] == f'{file_name}_{datetime.now().strftime("%Y_%m_%d")}_({i})'
    return assert_archive_file

def test_main(spark, create_supplier_po_item_df, create_gds_unspsc_report_df,
              get_table_storage_location_mocker, get_location_base_mocker,
              get_delta_table_mocker, get_batch_timestamp_range_mocker,
              current_test_dir, base_testdata_path, assertion):
    try:
        ml_supplier_po_item_gds_export_module.main()

        base_dir = os.path.normpath(current_test_dir + f"/{base_testdata_path}")
        export_file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        export_file_dir = f"{base_dir}/sap_export_"
        export_storage_location = f"{export_file_dir}gds/{export_file_name}_*"
        table_list = spark.read.format("csv").options(header='true').load(export_storage_location).collect()
        assert len(table_list) == 1
        assert len(table_list[0]) == 7
        assert table_list[0]["PROCESSED_DESCRIPTION"] == 'gg'

        report_table_storage_location = f"{base_dir}/gds_unspsc_report_df"
        report_table_list = spark.read.format("delta").load(report_table_storage_location).collect()
        desc_list=list(map(lambda gds_report:gds_report.PROCESSED_DESCRIPTION, report_table_list))
        assert len(report_table_list) == 4
        expect_desc_list=['dd', 'ee', 'ff', 'gg']
        assert len(desc_list) == len(expect_desc_list)
        assert all(item in expect_desc_list for item in desc_list)

        archive_dir = f"sap_export_gds_archive/"
        assert archive_dir not in list(map(lambda fileInfo:fileInfo.name, sbnutils.get_dbutils().fs.ls(base_dir)))

        file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        # archive file end with file_name
        ml_supplier_po_item_gds_export_module._mv_to_archive(f"{base_dir}/sap_export_gds")
        assertion(MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value, 1, base_dir)

        # archive file end with file_name_(1)
        create_supplier_po_item_df.write.mode("overwrite").option("header", "true").csv(f'{base_dir}/sap_export_gds/{file_name}_{datetime.now().strftime("%Y_%m_%d")}')
        ml_supplier_po_item_gds_export_module._mv_to_archive(f"{base_dir}/sap_export_gds")
        assertion(MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value, 2, base_dir)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

def test_read_data(create_supplier_po_item_df, get_table_storage_location_mocker, 
                   get_batch_timestamp_range_mocker, current_test_dir, base_testdata_path):
    read_dataframe = ml_supplier_po_item_gds_export_module._read_data(
        os.path.normpath(current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df")
    )
    df_list = read_dataframe.collect()
    desc_list=list(map(lambda po_item:po_item.PROCESSED_DESCRIPTION, df_list))
    assert len(df_list) == 2
    assert len(df_list[0]) == 7
    expect_desc_list=['dd', 'gg']
    assert len(desc_list)==len(expect_desc_list)
    assert all(item in expect_desc_list for item in desc_list)



    