import mock
import pytest
import numpy as np
from mock import patch
import pyspark
from pyspark.sql import SparkSession

from modules.utils.persist_and_load_utils import load_table, persist_to_delta_table, persist_to_numpy, load_numpy

########################################################################################
## Fixtures
######################################################################################## 

########################################################################################
## Function Tests
########################################################################################
def test_load_table():
    # Create a mock SparkSession object
    spark_mock = mock.Mock(spec=pyspark.sql.SparkSession)

    # Mock the return value of the load function
    spark_mock.read.csv.return_value = spark_mock
    spark_mock.read.format.return_value.load.return_value = spark_mock

    # Call the load_table function with mock arguments
    result = load_table("path/to/table", "csv", spark_mock)

    # Assert that the mock functions were called with the correct arguments
    spark_mock.read.csv.assert_called_once_with("path/to/table", header=True, inferSchema=True)
    assert result == spark_mock

    # Reset the mock and call load_table again with different table_type
    spark_mock.reset_mock()
    result = load_table("path/to/table", "delta", spark_mock)

    # Assert that the mock functions were called with the correct arguments
    spark_mock.read.format.assert_called_once_with("delta")
    spark_mock.read.format.return_value.load.assert_called_once_with("path/to/table")
    assert result == spark_mock

    # Reset the mock and call load_table again with unsupported table_type
    spark_mock.reset_mock()
    with pytest.raises(ValueError):
        result = load_table("path/to/table", "json", spark_mock)
        
def test_persist_to_delta_table(spark):
    # Define test data
    data = [("Alice", 1), ("Bob", 2), ("Charlie", 3)]
    df = spark.createDataFrame(data, ["name", "age"])

    # Define test arguments
    storage_path = "/tmp/test_delta"
    mode = "overwrite"
    database_name = "test_db"
    table_name = "test_table"

    with mock.patch("modules.utils.persist_and_load_utils.setup") as mock_setup, \
         mock.patch.object(pyspark.sql.DataFrameWriter, "save") as mock_save:
        
        # Call the function
        persist_to_delta_table(df, 
                               storage_path, 
                               mode, 
                               #database_name, 
                               #table_name, 
                               #spark
                              )

        # Check the Delta table was saved with the right arguments
        mock_save.assert_called_once_with(storage_path)
        
        # Check the Delta table was created in the database
        #mock_setup.assert_called_once_with(database_name, table_name, spark)
        #spark.sql.assert_called_once_with(f"CREATE TABLE {database_name}.{table_name} USING DELTA LOCATION '{storage_path}';")
    
def test_persist_to_numpy():
    # Define test data
    test_array = np.array([1, 2, 3, 4])

    # Define test arguments
    storage_path = "/tmp/test_array.npy"

    with mock.patch("builtins.open", mock.mock_open()) as mock_open, \
         mock.patch("numpy.save") as mock_save:
        # Call the function
        persist_to_numpy(test_array, storage_path)

        # Check that the file was opened with the right mode and path
        mock_open.assert_called_once_with(storage_path, "wb")

        # Check that the array was saved to the file
        mock_file_handle = mock_open()
        mock_save.assert_called_once_with(mock_file_handle, test_array)     

        