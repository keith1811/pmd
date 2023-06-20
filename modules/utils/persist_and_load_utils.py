from typing import Union

import pyspark
import numpy as np
import pandas as pd


def setup(database_name: str, table_name: str, spark: pyspark.sql.SparkSession) -> None:
    """
    Set up database to use. Create the database {database_name} if it doesn't exist, 
    and drop the table {table_name} if it exists

    Parameters
    ----------
    database_name: str
        Database to create if it doesn't exist. Otherwise use database of the name provided
    table_name: str
        Drop table if it already exists
    spark: pyspark.sql.SparkSession 
        The SparkSession object.
    """
    spark.sql(f'CREATE DATABASE IF NOT EXISTS {database_name};')
    spark.sql(f'USE {database_name};')
    spark.sql(f'DROP TABLE IF EXISTS {database_name}.{table_name};')

def load_table(table_path: str, table_type: Union[str, None], spark: pyspark.sql.SparkSession) -> pyspark.sql.DataFrame:
    """
    Read a table in CSV or Parquet format from a remote location and apply a column prefix if specified.

    Parameters
    -------
        table_path: str
            The path to the table on the remote location.
        table_type: Optional[str, None]
            The file format of the table, either "csv", "delta" or "parquet".
        spark: pyspark.sql.SparkSession
            A SparkSession object

    Returns:
    -------
        A DataFrame containing the data from the remote table.

    """
    # Validate the table type
    if table_type not in ["csv", "delta", "parquet", None]:
        raise ValueError(f"Unsupported table type: {table_type}. Supported types are csv, delta and parquet.")

    # Read the table based on its type
    if table_type == "csv":
        ingested_df = spark.read.csv(table_path, header=True, inferSchema=True)
    elif table_type == "delta":
        ingested_df = spark.read.format("delta").load(table_path)
    elif table_type == "parquet": 
        ingested_df = spark.read.format("parquet").load(table_path)
    elif not table_type:
        ingested_df = spark.read.format("delta").load(table_path)

    return ingested_df

def persist_to_delta_table(df: pyspark.sql.DataFrame,
                           storage_path: str,
                           mode: str = "overwrite",
                           database_name: str = None, 
                           table_name: str = None,  
                           spark: pyspark.sql.SparkSession = None
                           ) -> None:
    """
    Persists a Spark DataFrame to a Delta table in the specified database and storage location.

    Parameters
    ----------
    df: pyspark.sql.DataFrame 
        The DataFrame to be persisted.
    storage_path: str 
        The path to the storage location for the Delta table.
    mode: str
        mode to save data
    database_name: str 
        The name of the database to use. If it doesn't exist, it will be created.
    table_name: str 
        The name of the table to use. If it already exists, it will be dropped.
    spark: pyspark.sql.SparkSession
        The SparkSession object.
    """
    # Persist the DataFrame to Delta format
    df.write.format("delta").mode(mode).save(storage_path)
    
    # in case the config for unity catalog is provided
    if database_name and table_name and spark:
        # Create database if not exists, drop table if it already exists
        setup(database_name, table_name, spark)
        # Create a Delta table from the persisted data and register it in the specified database
        spark.sql(f"CREATE TABLE {database_name}.{table_name} USING DELTA LOCATION '{storage_path}';")

def persist_pandas_to_csv(df: pd.DataFrame,
                           storage_path: str) -> None:
    """
    Persists a Spark DataFrame to a Delta table in the specified database and storage location.

    Parameters
    ----------
    df: pd.DataFrame 
        The DataFrame to be persisted.    
    """
    
    df.to_csv(storage_path, header= True, index=False)      
        
        
def persist_to_numpy(numpy_array: np.ndarray, storage_path: str) -> None:
    """
    Stores a NumPy array to disk as a binary file.

    Parameters
    ----------
    numpy_array: numpy.ndarray
        The NumPy array to store.
    storage_path: str 
        The file path to store the array.
    """
    with open(storage_path, 'wb') as file:
        np.save(file, numpy_array)

def load_numpy(storage_path: str) -> np.ndarray:
    """
    Loads a NumPy array from a binary file.

    Parameters
    ----------
    storage_path: str
        The file path to load the array.

    Returns
    ----------
    numpy.ndarray: 
        The loaded NumPy array.

    """
    with open(storage_path, 'rb') as file:
        numpy_array = np.load(file)

    return numpy_array
