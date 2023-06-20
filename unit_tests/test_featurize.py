import pytest

import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.sql.types import DateType, IntegerType, StringType, StructField, StructType

from modules.utils.featurize import Featurizer
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

def spark_df_data_equal(df1, df2):
    """
    Check if two Spark dataframes are equal, regardless of order of rows.
    """
    df1_sorted = df1.sort(df1.columns)
    df2_sorted = df2.sort(df2.columns)
    return df1_sorted.collect() == df2_sorted.collect()

########################################################################################
## Fixtures
########################################################################################    

@pytest.fixture
def sample_dataframe(spark):
    """Mock sample df 1."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), True),
            StructField("BUYER_ORG", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [(None, "12345", "1", "KRONOS Enterprise Archive V8",       "001", "Material", "12345678", "12", "34", "56", "78"),
            ("88888", "34567", "Nan" ,"CHARGING IN PROGRESS SIGN",         "Null", "Material", "23456789", "23", "45", "67", "89"),
            ("77777", "45678", "2", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
            ("Not Available", "56789", "1", None,                         "002", "Material", "45678910", "45", "67", "89", "10"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand", "nan", "Material", "56789101", "56", "78", "91", "01"),
            ("77777", "45678", "6", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
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
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [("N/A", "12345", "1", "KRONOS Enterprise Archive V8",       "001", "Material", "12345678", "12", "34", "56", "78"),
            ("77777", "45678", "2", "Pump panel sign",                    "N/A", "Material", "34567891", "34", "56", "78", "91"),
            ("N/A",   "56789", "1", None,                                 "002", "Material", "45678910", "45", "67", "89", "10"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand", "N/A", "Material", "56789101", "56", "78", "91", "01"),
            ("77777", "45678", "6", "Pump panel sign",                    "N/A", "Material", "34567891", "34", "56", "78", "91"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def exp_output_df_2(spark):
    """Mock expected output df 2."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), True),
            StructField("BUYER_ORG", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [(None, "12345", "1", "KRONOS Enterprise Archive V8",       "001", "Material", "12345678", "12", "34", "56", "78"),
            ("88888", "34567", "Nan" ,"CHARGING IN PROGRESS SIGN",        "Null", "Material", "23456789", "23", "45", "67", "89"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand", "nan", "Material", "56789101", "56", "78", "91", "01"),
            ("77777", "45678", "2", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def exp_output_df_3(spark):
    """Mock expected output df 3."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), True),
            StructField("BUYER_ORG", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [(None, "12345", "1", "KRONOS Enterprise Archive V8",       "001", "Material", "12345678", "12", "34", "56", "78"),
            ("88888", "34567", "Nan" ,"CHARGING IN PROGRESS SIGN",         "Null", "Material", "23456789", "23", "45", "67", "89"),
            ("77777", "45678", "2", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand", "nan", "Material", "56789101", "56", "78", "91", "01"),
            ("77777", "45678", "6", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def exp_output_df_4(spark):
    """Mock expected output df 4."""
    schema = StructType(
        [
            StructField("SUPPLIER_ORG", StringType(), False),
            StructField("BUYER_ORG", StringType(), True),
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("DESCRIPTION", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("CODE", StringType(), True),
            StructField("SEGMENT", StringType(), True),
            StructField("FAMILY", StringType(), True),
            StructField("CLASS", StringType(), True),
            StructField("COMMODITY", StringType(), True)
        ]
    )

    data = [("N/A", "12345", "1", "KRONOS Enterprise Archive V8",       "001", "Material", "12345678", "12", "34", "56", "78"),
            ("88888", "34567", "Nan" ,"CHARGING IN PROGRESS SIGN",         "Null", "Material", "23456789", "23", "45", "67", "89"),
            ("77777", "45678", "2", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
            ("Not Available", "56789", "1", None,                         "002", "Material", "45678910", "45", "67", "89", "10"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand", "nan", "Material", "56789101", "56", "78", "91", "01"),
            ("77777", "45678", "6", "Pump panel sign",                    "?",   "Material", "34567891", "34", "56", "78", "91"),
    ]

    return spark.createDataFrame(data, schema)


@pytest.fixture
def select_col_instructions():
    select_cols_instructions = {"select": ["SUPPLIER_ORG", "BUYER_ORG"]}
    
    return select_cols_instructions

@pytest.fixture
def handle_str_na_instructions():
    handle_str_na_instructions = {"columns": ["SUPPLIER_ORG", "PO_ITEM_ID", "MANUFACTURER_PART"], 
                                  "na_vals_list": ["Not Available", "?", "N/A", "Nan","nan", "Null"], 
                                  "replacement_val": "N/A",
                                  "na_filter_instructions": {"not_contains_char_filter":{"PO_ITEM_ID": "N/A"}}
                                 }
    
    return handle_str_na_instructions

@pytest.fixture
def drop_duplicates_instructions():
    drop_duplicates_instructions = {"index_col": ["PO_ITEM_ID"], 
                                  "cat_cols": ["SUPPLIER_ORG", "BUYER_ORG", "DESCRIPTION", "MANUFACTURER_PART", "ITEM_TYPE", "CODE"],            
                                 }
    
    return drop_duplicates_instructions

@pytest.fixture
def dropna_config():
    dropna_config = {"subset": ["DESCRIPTION"], 
                     "how": "any"
                                 }
    return dropna_config

@pytest.fixture
def fillna_config():
    fillna_config = {"value": "N/A",
                     "subset": ["SUPPLIER_ORG"]
                                 }
    return fillna_config

########################################################################################
## Function Tests
########################################################################################
def test_select_columns(sample_dataframe, select_col_instructions):
    # define expected output
    expected_output = sample_dataframe.select("SUPPLIER_ORG", "BUYER_ORG")
    
    # call the method under test
    actual_output = Featurizer.select_columns(sample_dataframe, select_col_instructions)
    
    # check the output
    assert spark_df_equal(actual_output, expected_output)

def test_unify_and_filter_na_values(sample_dataframe, 
                                    handle_str_na_instructions, 
                                    exp_output_df_1):
    # call the method under test
    actual_output = Featurizer.unify_and_filter_na_values(sample_dataframe, handle_str_na_instructions)

    # check the output
    assert spark_df_data_equal(actual_output, exp_output_df_1)
    
def test_drop_duplicates(sample_dataframe, 
                         drop_duplicates_instructions, 
                         exp_output_df_2):
    # call the method under test
    actual_output = Featurizer.drop_duplicates(sample_dataframe, drop_duplicates_instructions)
    
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_2)    
    
def test_drop_missing_values(sample_dataframe, 
                             dropna_config, 
                             exp_output_df_3):
    # call the method under test
    actual_output = Featurizer.drop_missing_values(sample_dataframe, dropna_config)

    # check the output
    assert spark_df_equal(actual_output, exp_output_df_3)      


def test_fill_missing_values(sample_dataframe, 
                             fillna_config, 
                             exp_output_df_4):
    # call the method under test
    actual_output = Featurizer.fill_missing_values(sample_dataframe, fillna_config)
    
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_4)      
    
    
    