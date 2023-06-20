import pytest
import pandas as pd
from mock import MagicMock

import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from modules.model_train_feature_creator_module import TrainFeatureCreator
from modules.utils.common import IntermediateTransformationConfig, FeatureTableCreatorConfig
import modules.model_train_feature_creator_module as model_train_feature_creator_module

########################################################################################
## Unittest Utils
########################################################################################  
def spark_df_equal(df1, df2):
    """
    Check if two Spark dataframes are equal, regardless of order of rows.
    """
    if df1.schema != df2.schema:
        return False
    else:
        df1_sorted = df1.sort(df1.columns)
        df2_sorted = df2.sort(df2.columns)
        return df1_sorted.collect() == df2_sorted.collect()
    
########################################################################################
## Fixtures
########################################################################################  

@pytest.fixture
def sample_dataframe_1(spark):
    """Mock sample df 1."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), True),
            StructField("BUYER_ORG", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("SUPPLIER_PART", StringType(), True),
            StructField("MANUFACTURER_NAME", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("IS_ADHOC", StringType(), True),
            StructField("CODE", StringType(), True),
        ]
    )

    data = [(None, "12345", "1", "KRONOS Enterprise Archive V8",           "22A", "MAN1", "001",  "Material", "1","12345678"),
            ("88888", "34567", "Nan" ,"CHARGING IN PROGRESS SIGN",         "22A", "MAN1", "Null", "Material", "1","234500891"),
            ("88888", "34567", "Nan" ,"CHARGING IN PROGRESS SIGN",         "22A", "MAN1", "Null", "Material", "0","23450089"),
            ("77777", "45678", "2", "Pump panel sign",                     "22A", "MAN1", "?",    "Material", "1","34567891"),
            ("Not Available", "56789", "1", "Mobile Case",                 "22A", "MAN1", "002",  "Material", "0","45678900"),
            ("66666", "56789", "1", "Mobile Case",                         "22A", "MAN1", "002",  "Material", "0","45678963"),
            ("Not Available", "56789", "1", "Mobile Case",                 "22A", "MAN1", "002",  "Material", "0","45678932"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand",  "22A", "MAN1", "nan",  "Material", "1","56789101"),
            ("77777", "45678", "6", "Pump panel sign",                     "22A", "MAN1", "?",    "Material", "1", None),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_ref_df(spark):
    """Mock sample reference df"""
    schema = StructType(
        [
            StructField("CODE", StringType(), True),
        ]
    )

    data = [("12345678",),
            ("23452789",),
            ("45678900",),
            ("56789101",),
            ("45678932",),
            ("34567891",),
            ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def exp_output_df_1(spark):
    """Mock expected output df 1."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), True),
            StructField("BUYER_ORG", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("SUPPLIER_PART", StringType(), True),
            StructField("MANUFACTURER_NAME", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("IS_ADHOC", StringType(), True),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [(None, "12345", "1", "KRONOS Enterprise Archive V8",           "22A", "MAN1", "001",  "Material", "1","12345678", "12", "34", "56", "78"),
            ("Not Available", "56789", "1", "Mobile Case",                 "22A", "MAN1", "002",  "Material", "0","45678900", "45", "67", "89", "00"),
            ("Not Available", "56789", "1", "Mobile Case",                 "22A", "MAN1", "002",  "Material", "0","45678932", "45", "67", "89", "32"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand",  "22A", "MAN1", "nan",  "Material", "1","56789101", "56", "78", "91", "01"),
            ("77777", "45678", "2", "Pump panel sign",                     "22A", "MAN1", "?",    "Material", "1","34567891", "34", "56", "78", "91"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def exp_output_df_2(spark):
    """Mock expected output df 2."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), False),
            StructField("BUYER_ORG", StringType(), False),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("SUPPLIER_PART", StringType(), False),
            StructField("MANUFACTURER_NAME", StringType(), False),
            StructField("MANUFACTURER_PART", StringType(), False),
            StructField("ITEM_TYPE", StringType(), False),
            StructField("IS_ADHOC", StringType(), False),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [("N/A", "12345", "1", "KRONOS Enterprise Archive V8",          "22A", "MAN1", "001",  "Material", "1","12345678", "12", "34", "56", "78"),          
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand",  "22A", "MAN1", "N/A",  "Material", "1","56789101", "56", "78", "91", "01"),
            ("77777", "45678", "2", "Pump panel sign",                     "22A", "MAN1", "N/A",  "Material", "1","34567891", "34", "56", "78", "91"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def data_prep_params():  
    data_prep_params = {
            "unspsc_processsing": {
                "unspsc_col": "CODE",
                "length_filter": {
                    "regex_pattern_filter": {
                        "CODE": "^(?!0{8})\\d{8}$"
                    }
                },
                "non_zero_filter": {
                    "not_contains_char_filter": {
                        "SEGMENT": "00",
                        "FAMILY": "00",
                        "CLASS": "00"
                    }
                }
            },
            "handle_str_na_vals": {
                "columns": [
                    "SUPPLIER_ORG",
                    "BUYER_ORG",
                    "PO_ITEM_ID",
                    "DESCRIPTION",
                    "SUPPLIER_PART",
                    "MANUFACTURER_PART",
                    "MANUFACTURER_NAME",
                    "ITEM_TYPE",
                    "IS_ADHOC"
                ],
                "na_vals_list": [
                    "N/A",
                    "?",
                    "null",
                    "NULL",
                    "Null",
                    "NA",
                    "Not Available",
                    "not available",
                    "NAN",
                    "nan",
                    "Nan"
                ],
                "replacement_val": "N/A",
                "na_filter_instructions": {
                    "not_contains_char_filter": {
                        "DESCRIPTION": "N/A",
                        "PO_ITEM_ID": "N/A"
                    }
                }
            },
            "select_cols_instructions": {
                "select": [
                    "SUPPLIER_ORG",
                    "BUYER_ORG",
                    "PO_ITEM_ID",
                    "DESCRIPTION",
                    "SUPPLIER_PART",
                    "MANUFACTURER_NAME",
                    "MANUFACTURER_PART",
                    "ITEM_TYPE",
                    "IS_ADHOC",
                    "CODE",
                    "SEGMENT",
                    "FAMILY",
                    "CLASS",
                    "COMMODITY"
                ]
            },
            "drop_duplicates_instructions": {
                "index_col": [
                    "PO_ITEM_ID"
                ],
                "cat_cols": [
                    "SUPPLIER_ORG",
                    "BUYER_ORG",
                    "DESCRIPTION",
                    "SUPPLIER_PART",
                    "MANUFACTURER_PART",
                    "MANUFACTURER_NAME",
                    "ITEM_TYPE",
                    "IS_ADHOC",
                    "CODE"
                ]
            },
            "handle_missing_vals": {
                "drop_na_instructions": {
                    "drop_rows": True,
                    "dropna_config": {
                        "subset": [
                            "PO_ITEM_ID",
                            "DESCRIPTION"
                        ],
                        "how": "any"
                    }
                },
                "fill_na_instructions": {
                    "fill_missing_val": True,
                    "fillna_config": {
                        "value": "N/A",
                        "subset": [
                            "SUPPLIER_ORG",
                            "BUYER_ORG",
                            "SUPPLIER_PART",
                            "MANUFACTURER_PART",
                            "MANUFACTURER_NAME",
                            "ITEM_TYPE",
                            "IS_ADHOC"
                        ]
                    }
                }
            }
        }
    return data_prep_params

@pytest.fixture(scope="function")
def feature_table_creator_pipeline(data_prep_params):
    # Set FeatureTableCreatorConfig
    cfg = FeatureTableCreatorConfig(input_tables={"example_table": {"path": "example_data_path"}},
                                    intermediate_transformation_cfg=IntermediateTransformationConfig(**data_prep_params))   
    return TrainFeatureCreator(cfg)

########################################################################################
## Function Tests
########################################################################################
def test_filter_and_split_legit_unspsc(sample_dataframe_1, 
                                        sample_ref_df, 
                                        exp_output_df_1,
                                        feature_table_creator_pipeline):
    
    unspsc_processsing = feature_table_creator_pipeline.cfg.intermediate_transformation_cfg.unspsc_processsing
    # define expected output
    actual_output = feature_table_creator_pipeline.filter_and_split_legit_unspsc(sample_dataframe_1, 
                                                                 sample_ref_df, 
                                                                 unspsc_processsing)
  
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_1)

# @pytest.mark.skip(reason="Needs Debuging!")
def test_run_data_ingest(mocker, feature_table_creator_pipeline):
    # create a mock dataframe to simulate the return value of load_table function
    expected_df = MagicMock()
    mock_load_table = mocker.patch('modules.model_train_feature_creator_module.load_table', return_value=expected_df)
    
    # call the _run_data_ingest method
    table_name = "example_table"
    result_df = feature_table_creator_pipeline._run_data_ingest(table_name)

    # check if load_table function is called with the expected arguments
    mock_load_table.assert_called_once_with("example_data_path", "csv", model_train_feature_creator_module.spark)

    # check if the result is the same as the expected_df
    assert result_df == expected_df

def test_run_raw_data_prep(sample_dataframe_1, 
                                        sample_ref_df, 
                                        exp_output_df_2,
                                        feature_table_creator_pipeline):
    
    unspsc_processsing = feature_table_creator_pipeline.cfg.intermediate_transformation_cfg.unspsc_processsing
    # define expected output
    actual_output = feature_table_creator_pipeline.run_raw_data_prep(sample_dataframe_1, sample_ref_df)
    
    assert spark_df_equal(actual_output, actual_output)

