import os
import sys
from _decimal import Decimal

import mock
import pytest
from delta.tables import DeltaTable
from pyspark.sql.types import StructType, StringType, StructField, LongType, TimestampType, DecimalType
from utils.constants import Zone

from modules.utils.constants import MLTable

sys.path.append("../modules")
import ml_supplier_po_item_gds_update_module

from datetime import datetime, timedelta

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
            StructField("DESCRIPTION", StringType(), True),
            StructField("PROCESSED_DESCRIPTION", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_SEGMENT", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_FAMILY", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_CLASS", StringType(), True),
            StructField("EXTERNAL_PREDICATED_UNSPSC_COMMODITY", StringType(), True),
            StructField("EXTERNAL_PREDICATION_LASTUPDATED_AT", TimestampType(), True),
            StructField("_DELTA_CREATED_ON", TimestampType(), True),
            StructField("_DELTA_UPDATED_ON", TimestampType(), True)
        ]
    )

    df = spark.createDataFrame([
        ("1", 1001, 'aa', 'aa', None, None, None, None, None, timestamp_before_earliest, timestamp_before_earliest),
        ("2", 1002, 'BB', 'bb', None, None, None, None, None, timestamp_before_earliest, timestamp_before_earliest),
        ("3", 1004, 'cc', 'cc', None, None, None, None, None, timestamp_before_earliest, timestamp_before_earliest),
        ("4", 1003, 'cc4', 'cc', None, None, None, None, None, timestamp_before_earliest, timestamp_before_earliest),
        ("5", 1005, 'dd', 'dd', '12', '1234', '123456', '12345678', timestamp_before_earliest, timestamp_current,
         timestamp_current),
        ("6", 1006, 'dd4,|', 'dd', None, None, None, None, None, timestamp_current, timestamp_current),
        ("7", 1007, 'ee', 'ee', '12', None, None, None, None, timestamp_current, timestamp_current),
        ("8", 1008, 'ff', 'ff', '', None, None, None, None, timestamp_current, timestamp_current),
        ("9", 1009, 'g///@g', 'gg', None, None, None, None, None, timestamp_current, timestamp_current),
        ("10", 1010, 'hh', 'hh', None, None, None, None, None, timestamp_current, timestamp_current),
        ("11", 1010, 'kk', 'kk', None, None, None, '', None, timestamp_current, timestamp_current),
    ], schema)
    df.coalesce(1).write.format("delta").mode('overwrite').option("header", "true").save(
        os.path.normpath(current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df"))
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
            StructField("CONFIDENCE_SEGMENT", DecimalType(6, 5), True),
            StructField("CONFIDENCE_FAMILY", DecimalType(6, 5), True),
            StructField("CONFIDENCE_CLASS", DecimalType(6, 5), True),
            StructField("CONFIDENCE_COMMODITY", DecimalType(6, 5), True),
            StructField("REPORT_LASTUPDATED_AT", TimestampType(), True),
            StructField("_DELTA_CREATED_ON", TimestampType(), True),
            StructField("_DELTA_UPDATED_ON", TimestampType(), True)
        ]
    )

    spark.createDataFrame([
        ('d/d', 'dd', '12', '1234', '123456', '12345678', *(Decimal(0.78900) for _ in range(4)), timestamp_current,
         timestamp_current, timestamp_current),
        ('e@e', 'ee', '12', '1234', '123456', '12345678', *(Decimal(0.78900) for _ in range(4)), timestamp_current,
         timestamp_current, timestamp_current),
        ('ff ', 'ff', '12', '1234', '123456', '12345678', *(Decimal(0.78900) for _ in range(4)), timestamp_current,
         timestamp_current, timestamp_current)
    ], schema).coalesce(1).write.format("delta").mode('overwrite').option("overwriteSchema", "true").option("header",
                                                                                                            "true") \
        .save(os.path.normpath(current_test_dir + f"/{base_testdata_path}/gds_unspsc_report_df"))


@pytest.fixture
def get_table_storage_location_mocker(current_test_dir, base_testdata_path):
    with mock.patch("utils.sbnutils.get_table_storage_location") as table_storage_location:
        values = {(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value): os.path.normpath(
            current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df"),
                  (Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value): os.path.normpath(
                      current_test_dir + f"/{base_testdata_path}/gds_unspsc_report_df")}
        table_storage_location.side_effect = mock.Mock(side_effect=lambda x, y: values[(x, y)])
        yield table_storage_location


@pytest.fixture
def get_delta_table_mocker(spark, current_test_dir, base_testdata_path):
    with mock.patch("utils.sbnutils.get_delta_table") as delta_table:
        values = {(Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value): DeltaTable.forPath(
            spark, current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df"),
            (Zone.ENRICHMENT.value, MLTable.ENRICHMENT_ML_GDS_UNSPSC_REPORT.value): DeltaTable.forPath(
                spark, current_test_dir + f"/{base_testdata_path}/gds_unspsc_report_df")}
        delta_table.side_effect = mock.Mock(side_effect=lambda x, y: values[(x, y)])
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
def fix_po_item_columns():
    columns = [
        "ID",
        "PO_ITEM_ID",
        "PROCESSED_DESCRIPTION",
        "EXTERNAL_PREDICATED_UNSPSC_SEGMENT",
        "EXTERNAL_PREDICATED_UNSPSC_FAMILY",
        "EXTERNAL_PREDICATED_UNSPSC_CLASS",
        "EXTERNAL_PREDICATED_UNSPSC_COMMODITY",
        "EXTERNAL_PREDICATION_LASTUPDATED_AT"]
    return columns


def test_read_data(create_supplier_po_item_df, get_table_storage_location_mocker,
                   get_batch_timestamp_range_mocker, current_test_dir, base_testdata_path, fix_po_item_columns):
    read_dataframe = ml_supplier_po_item_gds_update_module._read_data(
        os.path.normpath(current_test_dir + f"/{base_testdata_path}/ml_supplier_po_item_df"), fix_po_item_columns,
        'delta')
    df_list = read_dataframe.collect()
    desc_list = list(map(lambda po_item: po_item.PROCESSED_DESCRIPTION, df_list))
    assert len(df_list) == 6
    assert len(df_list[0]) == 8
    expect_desc_list = ['dd', 'ee', 'ff', 'gg', 'hh', 'kk']
    assert len(desc_list) == len(expect_desc_list)
    assert all(item in expect_desc_list for item in desc_list)


def test_main(spark, create_supplier_po_item_df, create_gds_unspsc_report_df,
              get_table_storage_location_mocker, get_delta_table_mocker, get_batch_timestamp_range_mocker,
              current_test_dir, base_testdata_path):
    try:
        ml_supplier_po_item_gds_update_module.main()

        base_dir = os.path.normpath(current_test_dir + f"/{base_testdata_path}")
        update_file_name = MLTable.ENRICHMENT_ML_SUPPLIER_PO_ITEM.value
        update_storage_location = f"{base_dir}/{update_file_name}_df"
        table_list = spark.read.format("delta").options(header='true').load(update_storage_location).collect()
        assert len(table_list) == 11
        assert len(table_list[0]) == 11
        assert table_list[0]["PROCESSED_DESCRIPTION"] == 'aa'
        assert table_list[0]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] is None
        assert table_list[5]["EXTERNAL_PREDICATED_UNSPSC_SEGMENT"] == '12'

        report_table_storage_location = f"{base_dir}/gds_unspsc_report_df"
        report_table_list = spark.read.format("delta").load(report_table_storage_location).collect()
        desc_list = list(map(lambda gds_report: gds_report.PROCESSED_DESCRIPTION, report_table_list))
        assert len(report_table_list) == 3
        assert report_table_list[0]["PROCESSED_DESCRIPTION"] == 'dd'
        assert report_table_list[0]["UNSPSC_SEGMENT"] == '12'
        expect_desc_list = ['dd', 'ee', 'ff']
        assert len(desc_list) == len(expect_desc_list)
        assert all(item in expect_desc_list for item in desc_list)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")
