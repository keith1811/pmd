import pytest
import pandas as pd

import pyspark.sql.functions as f

from modules.utils.spark_utils import *

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
    df1_sorted = df1.sort(df1.columns)
    df2_sorted = df2.sort(df2.columns)
    return df1_sorted.collect() == df2_sorted.collect()

########################################################################################
## Fixtures
########################################################################################    

@pytest.fixture
def sample_dataframe_1(spark):
    """Mock sample df 1."""
    return spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"])

@pytest.fixture
def sample_dataframe_2(spark):
    """Mock sample df 2."""
    return [
        spark.createDataFrame([(1, "A"), (2, "B")], ["id", "value"]),
        spark.createDataFrame([(3, "C"), (4, "D")], ["id", "value"])
    ]

@pytest.fixture
def sample_dataframe_3(spark):
    """Mock sample df 3."""
    return spark.createDataFrame(
        [
            ("jane", 26, "female"),
            ("jake", 22, "male"),
            ("john", 30, None),
            ("jill", 33, "female"),
            ("jim", 28, "male"),
            ("mark", 25, "male"),
            ("alex", 28, "male"),
            ("markus", 23, "male"),
        ],
        ["name", "age", "gender"],
    )

@pytest.fixture
def sample_dataframe_4(spark):
    """Mock sample df 4."""
    table_l = spark.createDataFrame(
        [
            ("1", "A"),
            ("2", "B"),
            ("3", "C"),
            ("4", "D"),
            ("5", "E"),
        ],
        ["id", "value_l"],
    )
    table_r = spark.createDataFrame(
        [("2", "X"), ("4", "Y")], ["id", "value_r"]
    )
    return table_l, table_r    

@pytest.fixture
def sample_dataframe_5(spark):
    """Mock sample df 5."""
    return spark.createDataFrame(
        [
            ("John", "Doe", "us"), 
            ("Jane", "Doe", "ge"),
            ("John", "Rabi", "au"),
            ("Jane", "Smith", "us"),
            ("Jim", "Jones", "ge"),
            ("Jack", "Smith", "en"),
            ("Jill", "Jones", "ge"),
            ("John", "Travis", "us"),
            ("Jane", "Cruise", "us")
        ],
        ["first_name", "last_name", "country"]
    )

@pytest.fixture
def exp_output_df_1(sample_dataframe_1):
    return sample_dataframe_1.withColumnRenamed("id", "test_prefix_id").withColumnRenamed("value", "test_prefix_value")
    
@pytest.fixture
def exp_output_df_2_distinct(spark):
    return spark.createDataFrame([(1, "A"), (2, "B"), (3, "C"), (4, "D")], ["id", "value"]).distinct()

@pytest.fixture
def exp_output_df_2_no_distinct(spark):
    return spark.createDataFrame([(1, "A"), (2, "B"), (3, "C"), (4, "D")], ["id", "value"])

@pytest.fixture
def exp_output_df_3(spark):
    return spark.createDataFrame(
        [("mark", "male"), ("markus", "male")], ["first_name", "sex"]
    )   

@pytest.fixture
def exp_output_df_4(spark):
    return spark.createDataFrame(
        [("2", "B"), ("4", "D")], ["id", "value_l"]
    )    
    
@pytest.fixture
def exp_output_df_5(spark):
    return spark.createDataFrame(
        [
            ("jane", 26, "N/A"),
            ("jake", 22, "N/A"),
            ("john", 30, "N/A"),
            ("jill", 33, "N/A"),
            ("jim", 28, "N/A"),
            ("mark", 25, "N/A"),
            ("alex", 28, "N/A"),
            ("markus", 23, "N/A"),
        ],
        ["name", "age", "gender"],
    )

@pytest.fixture
def exp_output_df_6(spark):
    updated_df =  spark.createDataFrame(
            [
            ("John", "Doe", "us"), 
            ("Jane", "Doe", "ge"),
            ("John", "Rabi", "rare_country"),
            ("Jane", "Smith", "us"),
            ("Jim", "Jones", "ge"),
            ("Jack", "Smith", "rare_country"),
            ("Jill", "Jones", "ge"),
            ("John", "Travis", "us"),
            ("Jane", "Cruise", "us")
            ],
        ["first_name", "last_name", "country"]
    )
    
    infrequent_df = spark.createDataFrame([("au",),("en",)],
                                          ["country"])
    return  updated_df, infrequent_df

@pytest.fixture
def exp_output_df_7():
    data = {"country":["us", "ge", "au", "en"], "count": [4, 3, 1, 1]}
    
    return pd.DataFrame(data).set_index("country")

@pytest.fixture
def exp_output_df_8(spark):
    updated_df =  spark.createDataFrame(
            [
            ("John", "Doe", "us"), 
            ("Jane", "Doe", "ge"),
            ("Jane", "Smith", "us"),
            ("Jim", "Jones", "ge"),
            ("Jill", "Jones", "ge"),
            ("John", "Travis", "us"),
            ("Jane", "Cruise", "us")
            ],
        ["first_name", "last_name", "country"]
    )
    
    infrequent_df = pd.DataFrame({"country":["au","en"]})
    return  updated_df, infrequent_df




@pytest.fixture
def filter_instructions():
    filter_instructions = {
        "filter": {"gender": "male"},
        "ne_filter": {"name": ["john"]},
        "not_contains_char_filter": {"name": "ill"},
        "regex_pattern_filter": {"name": "ma[a-z]"},
        "sort_values": {"sort_columns": ["name"], "is_ascending": True},
        "rename": {"name": "first_name", "gender": "sex"},
        "select": ["first_name", "sex"],
    }
    return filter_instructions

########################################################################################
## Function Tests
########################################################################################
def test_add_prefix_to_columns(sample_dataframe_1, exp_output_df_1):
    actual_output = add_prefix_to_columns(sample_dataframe_1, "test_prefix")
    
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_1)    
    
def test_union_dfs_distinct(sample_dataframe_2, exp_output_df_2_distinct):
    actual_output = union_dfs(*sample_dataframe_2, distinct=True)
    
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_2_distinct)

def test_union_dfs_no_distinct(sample_dataframe_2, exp_output_df_2_no_distinct):
    actual_output = union_dfs(*sample_dataframe_2, distinct=False)
    
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_2_no_distinct)   
    
def test_filter_df_from_instructions(sample_dataframe_3, exp_output_df_3, filter_instructions):
 
    actual_output = filter_df_from_instructions(sample_dataframe_3, filter_instructions)    
    
    # check the output
    assert spark_df_equal(actual_output, exp_output_df_3) 

def test_leftsemi_filtering(sample_dataframe_4, exp_output_df_4):
    table_l, table_r = sample_dataframe_4
    filtered_table_l = leftsemi_filtering(table_l, table_r, "id", "id")
    
    # check the output
    assert spark_df_equal(filtered_table_l, exp_output_df_4) 

def test_replace_columns_values(sample_dataframe_3, exp_output_df_5):
    columns = "gender"
    values_list = ["male", "female"]
    replacement_value = "N/A"
    
    actual_output_df = replace_columns_values(sample_dataframe_3, columns, values_list, replacement_value)
    
    # check the output
    assert spark_df_data_equal(actual_output_df, exp_output_df_5) 

def test_replace_infrequent_values(sample_dataframe_5, exp_output_df_6):
    column = "country"
    threshold = 3
    replacement_value = "rare_country"
    
    expected_updated_df, expected_infrequent_values_df  = exp_output_df_6
    
    actual_updated_df, actual_infrequent_values_df = replace_infrequent_values(sample_dataframe_5, column, threshold, replacement_value)
    
    # check the outputs
    assert spark_df_data_equal(actual_updated_df, expected_updated_df)
    assert spark_df_data_equal(actual_infrequent_values_df , expected_infrequent_values_df) 

def test_count_unique_values(sample_dataframe_5, exp_output_df_7):
    actual_output = count_unique_values(sample_dataframe_5, "country")

    # check the outputs
    pd.testing.assert_frame_equal(actual_output, exp_output_df_7)     

def test_filter_by_unique_values_count(sample_dataframe_5, exp_output_df_8):
    filter_column = "country"
    filter_value = 3
    
    expected_updated_df, expected_infrequent_values_df  = exp_output_df_8
    
    actual_updated_df, actual_infrequent_values_df = filter_by_unique_values_count(sample_dataframe_5, "country", 3)
    
    # check the outputs
    assert spark_df_equal(actual_updated_df, expected_updated_df)
    pd.testing.assert_frame_equal(actual_infrequent_values_df, expected_infrequent_values_df) 
     