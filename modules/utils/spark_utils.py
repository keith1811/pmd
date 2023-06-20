from typing import Any, Dict, List, Union, Tuple

import pyspark
import pandas as pd
from functools import reduce
from pyspark.sql import functions as f


def add_prefix_to_columns(df: pyspark.sql.dataframe.DataFrame, prefix: str):
    """
    Add prefix the columns in the given DataFrame using the given table name as a prefix.

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        The DataFrame whose columns will be added a prefix.
    prefix: str
        prefix to be added to column (usually is the initials of the table name)

    Returns
    ----------
    pyspark.sql.dataframe.DataFrame: The modified DataFrame with added prefix.
    """
    for column in df.columns:
        df = df.withColumnRenamed(column, prefix + "_" + column)
    return df

def union_dfs(*input_dfs: pyspark.sql.DataFrame, 
              distinct: bool = True) -> pyspark.sql.DataFrame:
    """
    Takes any number of dataframes
    with the same schema and unions them by column names

    Parameters
    ----------
        *input_dfs: List[pyspark.sql.DataFrame]
            List of loaded pyspark dataframes
        distinct: Bool
            Bool to indicate whether the output should be a distinct DataFrame

    Returns
    -------
    pyspark.sql.DataFrame
        Unioned Spark DataFrame      
    """
    if distinct:
        return reduce(pyspark.sql.DataFrame.unionByName, input_dfs).distinct() 

    return reduce(pyspark.sql.DataFrame.unionByName, input_dfs)

def filter_df_from_instructions(
    df: pyspark.sql.dataframe.DataFrame, params_instruction: Dict[Any, Any],
) -> pyspark.sql.dataframe.DataFrame:
    """
    Filter table given instruction parameters.

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        given table
    params_instruction: Dict[Any, Any]
        instructions.

    Returns
    ----------
    pyspark.sql.dataframe.DataFrame: A filtered dataframe.
    """

    filtered_df = df

    filtered_df = filter_func(params_instruction, filtered_df)
    
    filtered_df = regex_pattern_filter_func(params_instruction, filtered_df)
            
    filtered_df = ne_filter_func(params_instruction, filtered_df)

    filtered_df = contains_char_filter_func(params_instruction, filtered_df)

    filtered_df = not_contains_char_filter_func(params_instruction, filtered_df)
    
    filtered_df = sort_values_func(params_instruction, filtered_df)

    filtered_df = rename_func(params_instruction, filtered_df)

    filtered_df = select_func(params_instruction, filtered_df)

    return filtered_df

def select_func(params_instruction, filtered_df):
    if "select" in params_instruction.keys():
        select_cols = params_instruction["select"]
        filtered_df = filtered_df.select(*select_cols)
    return filtered_df

def rename_func(params_instruction, filtered_df):
    if "rename" in params_instruction.keys():
        rename_dict = params_instruction["rename"]
        for current_col_name, desired_col_name in rename_dict.items():
            filtered_df = filtered_df.withColumnRenamed(
                current_col_name, desired_col_name
            )
            
    return filtered_df

def sort_values_func(params_instruction, filtered_df):
    if "sort_values" in params_instruction.keys():
        sort_instructions = params_instruction["sort_values"]
        filtered_df = filtered_df.sort(sort_instructions["sort_columns"], 
                                       ascending= sort_instructions["is_ascending"])
                                       
    return filtered_df

def not_contains_char_filter_func(params_instruction, filtered_df):
    if "not_contains_char_filter" in params_instruction.keys():
        filters = params_instruction["not_contains_char_filter"]
        for column, value in filters.items():
            filtered_df = filtered_df.filter(~f.col(column).contains(value))
    return filtered_df

def contains_char_filter_func(params_instruction, filtered_df):
    if "contains_char_filter" in params_instruction.keys():
        filters = params_instruction["contains_char_filter"]
        for column, value in filters.items():
            filtered_df = filtered_df.filter(f.col(column).contains(value))
    return filtered_df

def ne_filter_func(params_instruction, filtered_df):
    if "ne_filter" in params_instruction.keys():
        filters = params_instruction["ne_filter"]
        for column, value in filters.items():
            if isinstance(value, list):
                filtered_df = filtered_df.filter(~f.col(column).isin(value))
    return filtered_df

def regex_pattern_filter_func(params_instruction, filtered_df):
    if "regex_pattern_filter" in params_instruction.keys():
        filters = params_instruction["regex_pattern_filter"]
        for column, value in filters.items():
            filtered_df = filtered_df.filter(f.col(column).rlike(value))
    return filtered_df

def filter_func(params_instruction, filtered_df):
    if "filter" in params_instruction.keys():
        filters = params_instruction["filter"]
        for column, value in filters.items():
            if isinstance(value, list):
                filtered_df = filtered_df.filter(f.col(column).isin(value))
            elif value is None:
                filtered_df = filtered_df.filter(f.col(column).isNull())
            elif value == "NOT NULL":
                filtered_df = filtered_df.filter(f.col(column).isNotNull())
            else:
                filtered_df = filtered_df.filter("{} = '{}'".format(column, value))
    return filtered_df

def leftsemi_filtering(table_l: pyspark.sql.dataframe.DataFrame,
                       table_r: pyspark.sql.dataframe.DataFrame,
                       left_on: str,
                       right_on: str) -> pyspark.sql.dataframe.DataFrame:
    """
    Filters left table using a semi join on right table
    
    Parameters
    ----------
    table_l: pyspark.sql.dataframe.DataFrame
        Left table to join
    table_r: pyspark.sql.dataframe.DataFrame
        Right table to join
    left_on: str
        Join key of table1
    right_on: str
        Join key of table2

    Returns
    ----------
    pyspark.sql.dataframe.DataFrame
        Filtered left table
    """
    join_cnd = f.col(f"t1.{left_on}") == f.col(f"t2.{right_on}")
    filtered_left_table = table_l.alias("t1").join(table_r.alias("t2"), on=join_cnd, how="leftsemi")
    
    return filtered_left_table

def replace_columns_values(df: pyspark.sql.dataframe.DataFrame, 
                           columns: Union[list, str], 
                           values_list: list, 
                           replacement_value: Any) -> pyspark.sql.dataframe.DataFrame:
    """
    Replace selected values table, given instruction parameters.

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        given table
    columns: Union[list, str]
        given columns.
    values_list: list
        list of values to be searched for in rows
    replacement_value: str
        value to be replaced
    
    Returns
    ----------
    pyspark.sql.dataframe.DataFrame: A dataframe with replaced values 
    """
    if isinstance(columns, list):
        for col_name in columns:
            df = df.fillna("N/A", col_name)
            df = df.withColumn(col_name, f.when(df[col_name].isin(values_list), 
                                                replacement_value).otherwise(df[col_name]))
            
    else:
        df = df.fillna("N/A", columns)   
        df = df.withColumn(columns, f.when(df[columns].isin(values_list), replacement_value).otherwise(df[columns]))
        
    return df

def replace_infrequent_values(df: pyspark.sql.dataframe.DataFrame, 
                              column_name: str, 
                              threshold: int, 
                              replacement_value: str) -> Tuple[pyspark.sql.DataFrame, 
                                                               pyspark.sql.DataFrame]:
    """
    Counts the unique values in a specified column of a PySpark DataFrame, replaces values that occurs 
    less than a given threshold with a specified replacement value, and returns the updated DataFrame.

    Parameters
    ----------
    df: pyspark.sql.dataframe.DataFrame
        The PySpark DataFrame to update
    column_name: str 
        The name of the column to process
    threshold: int
        The minimum number of occurrences for a value to be retained
    replacement_value: str 
        The value to replace infrequent values with

    Returns
    -------
    pyspark.sql.dataframe.DataFrame: 
        The updated PySpark DataFrame with replaced value in the given column
    """
    # Get a count of each unique value in the specified column
    value_counts = df.groupBy(column_name).count()

    # Filter out values whose count is above the threshold
    infrequent_values_df = value_counts.filter(f.col("count") < threshold).select(column_name)
    infrequent_values_list = [row[column_name] for row in infrequent_values_df.collect()]
    
    # Replace the rare values in the DataFrame with a string indicating that they are rare
    df = replace_columns_values(df, column_name, infrequent_values_list, replacement_value)

    return df, infrequent_values_df

def count_unique_values(df: pyspark.sql.dataframe.DataFrame, count_column: str) -> pd.DataFrame:
    """
    Group a column by its unique values and count the number of occurrences of each value.

    parameters
    ----------
        df: pyspark.sql.dataframe.DataFrame
            The input PySpark DataFrame
        count_column: str
            The name of the column to group and count values by

    Returns
    -------
    pd.DataFrame: 
        A Pandas DataFrame with a count of the occurrences of each unique value
    """
    # Group the column by its unique values and count the number of occurrences of each value
    value_counts = df.select(count_column).groupBy(count_column).count().orderBy(f.desc_nulls_first("count"))

    # Collect the value counts data into the driver program as a list of tuples
    value_counts_df = pd.DataFrame(value_counts.collect(), columns = [count_column,"count"])
    value_counts_df.set_index(count_column, inplace= True)
    
    return value_counts_df

def filter_by_unique_values_count(df: pyspark.sql.dataframe.DataFrame, 
                                  filter_column: str, 
                                  filter_val: int) -> Tuple[pyspark.sql.dataframe.DataFrame, pd.DataFrame]:
    """
    Filter a PySpark DataFrame by unique value counts in a given column.

    Parameters
    ----------
        df: pyspark.sql.dataframe.DataFrame
            The input PySpark DataFrame
        filter_column: str
            The name of the column to filter values by
        filter_val: int
            The minimum count of the unique values to include in the filtered DataFrame

    Returns
    -------
    tuple[pyspark.sql.dataframe.DataFrame, pd.DataFrame]
        A tuple of Spark  and pandas DataFrames - the filtered DataFrame and the DataFrame of infrequent values.
    """
    value_counts_df = count_unique_values(df, filter_column)

    filtered_df = df.filter(f.col(filter_column).isin(list(value_counts_df[value_counts_df["count"] >= filter_val].index)))
    infrequent_valus_df = pd.DataFrame(value_counts_df[value_counts_df["count"] < filter_val].index)
    
    return filtered_df, infrequent_valus_df
