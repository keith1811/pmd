import mlflow
import pytest
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from mock import MagicMock, patch, Mock
import mock
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
    
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql.types import (DateType, IntegerType, StringType, 
                               StructField, StructType)

from modules.model_train_module import ModelTrain
from modules.utils.common import (ModelTrainConfig, MLflowTrackingConfig, 
                            IntermediateTableConfig, FeatureTransformationConfig)

from modules.utils.notebook_utils import load_and_set_env_vars, load_config


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
## Load training config and create training config objects
########################################################################################     
@pytest.fixture
def setup_train():

    # Set pipeline name
    pipeline_name = 'model_train'

    # Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
    pipeline_config = load_config(pipeline_name, "ml")

    # Load and set arbitrary params via spark_env_vars
    # Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
    env_vars = load_and_set_env_vars("ml")

    # Set MLflowTrackingFlow
    mlflow_tracking_cfg = MLflowTrackingConfig(run_name=pipeline_config["mlflow_params"]["run_name"],
                                            model_name=pipeline_config["mlflow_params"]["model_name"],
                                            registry_uri=mlflow.get_registry_uri())
    
    # Get Intermediate storage path
    intermediate_table_cfg = IntermediateTableConfig(storage_path=env_vars["intermediate_table_storage_path"])

    # Set feature transformation config
    feature_transformation_cfg = FeatureTransformationConfig(**pipeline_config["feature_transformation"])

    # Set ModelTrainConfig
    global cfg
    cfg = ModelTrainConfig(mlflow_tracking_cfg = mlflow_tracking_cfg,
                        intermediate_table_cfg = intermediate_table_cfg,
                        feature_transformation_cfg = feature_transformation_cfg,
                        train_val_split_config = pipeline_config["train_val_split_config"],
                        train_instructions = pipeline_config["train_instructions"],
                        pipeline_config = pipeline_config)    

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

    data = [("N/A", "12345", "1", "KRONOS Enterprise Archive V8",          "22A", "MAN1", "001", "Material", "1","12345678"), 
            ("88888", "34567", "N/A" ,"CHARGING IN PROGRESS SIGN",         "22A", "MAN1", "N/A", "Material", "0","23450089"),
            ("77777", "45678", "2", "Pump panel sign",                     "22A", "MAN1", "N/A", "Material", "1","12345678"),
            ("N/A", "56789", "1", "Mobile Case",                           "22A", "MAN1", "002", "Material", "0","12345678"),
            ("66666", "56789", "1", "Mobile Case",                         "22A", "MAN1", "002", "Material", "0","23450089"),
            ("N/A", "56789", "1", "Mobile Case",                           "22A", "MAN1", "002", "Material", "0","78965432"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand",  "22A", "MAN1", "N/A", "Material", "1","56789101"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def sample_dataframe_2():
    """Mock sample df 2.""" 
    return pd.DataFrame({
                        "CONCAT_FEATURE": ['this is a test', 'another test', 'and one more'],
                        "CODE": [0, 1, 0]})
@pytest.fixture
def sample_data_3():
    """Mock sample 3."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 1, 0, 1]) 
    
    return X, y

@pytest.fixture
def exp_labels():
    exp_labels = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ])
    
    return exp_labels
    
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

        ]
        )

    data = [("N/A", "12345", "1", "KRONOS Enterprise Archive V8",          "22A", "MAN1", "001", "Material", "1", "12345678"), 
            ("88888", "34567", "N/A" ,"CHARGING IN PROGRESS SIGN",         "22A", "MAN1", "N/A", "Material", "0", "23450089"),
            ("77777", "45678", "2", "Pump panel sign",                     "22A", "MAN1", "N/A", "Material", "1", "12345678"),
            ("N/A", "56789", "1", "Mobile Case",                           "22A", "MAN1", "002", "Material", "0", "12345678"),
            ("66666", "56789", "1", "Mobile Case",                         "22A", "MAN1", "002", "Material", "0", "23450089"),
            ("N/A", "56789", "1", "Mobile Case",                           "22A", "MAN1", "002", "Material", "0", "99999999"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand",  "22A", "MAN1", "N/A", "Material", "1", "99999999"),
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
            StructField("SUPPLIER_PART", StringType(), True),
            StructField("MANUFACTURER_NAME", StringType(), True),
            StructField("MANUFACTURER_PART", StringType(), True),
            StructField("ITEM_TYPE", StringType(), True),
            StructField("IS_ADHOC", StringType(), True),
            StructField("CODE", StringType(), True),
        ]
    )

    data = [("N/A", "12345", "1", "KRONOS Enterprise Archive V8",          "22A", "MAN1", "001", "Material", "1", "12345678"), 
            ("88888", "34567", "N/A" ,"CHARGING IN PROGRESS SIGN",         "22A", "MAN1", "N/A", "Material", "0", "23450089"),
            ("77777", "45678", "2", "Pump panel sign",                     "22A", "MAN1", "N/A", "Material", "1", "12345678"),
            ("N/A", "56789", "1", "Mobile Case",                           "22A", "MAN1", "002", "Material", "0", "12345678"),
            ("66666", "56789", "1", "Mobile Case",                         "22A", "MAN1", "002", "Material", "0", "23450089"),
    ]

    return spark.createDataFrame(data, schema)

@pytest.fixture
def exp_output_3():
    exp_output = [[2, 3, 4, 1, 0], [5, 1, 0, 0, 0], [6, 7, 8, 0, 0]]
    return np.array(exp_output)

@pytest.fixture
def vocab_size():
    return 100
    
@pytest.fixture
def max_text_len():
    return 5

@pytest.fixture
def num_classes():
    return 4

@pytest.fixture
def validation_size():
    return 0.2

@pytest.fixture
def model_train_mocked_timestamp(setup_train, monkeypatch, base_artifact_path):
    # Use monkeypatch to replace get_current_timestamp with a mock object
    timestamp_mock_method = Mock(return_value="2000-01-01_00-00-00")
    monkeypatch.setattr("modules.model_train_module.get_current_timestamp", timestamp_mock_method)

    # Use monkeypatch to set the value of BASE_ARTIFACT_PATH to base_artifact_path
    monkeypatch.setattr("modules.model_train_module.ModelTrain.BASE_ARTIFACT_PATH", base_artifact_path)
    # Use monkeypatch to set the value of BASE_ARTIFACT_PATH to base_artifact_path
    monkeypatch.setattr("modules.model_train_module.ModelTrain.FEATURES_STORAGE_PATH", base_artifact_path)

    # Initiate train class
    model_train = ModelTrain(cfg)
    yield model_train
        
@pytest.fixture
def common_objs_wrapper(dbutils, base_artifact_path, model_train_mocked_timestamp):
    return dbutils, base_artifact_path, model_train_mocked_timestamp

########################################################################################
## Function Tests
########################################################################################
 
def test_create_unique_dbfs_storage_dir(common_objs_wrapper):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper
    
    # Call the function with a base_path argument
    actual_output = model_train_mocked_timestamp._create_unique_dbfs_storage_dir(dbutils, 
                                                                                 base_artifact_path)
       
    # Check that the result is a string and starts with "tmp/training/"
    assert isinstance(actual_output, str)
    assert actual_output.startswith(model_train_mocked_timestamp.FEATURES_STORAGE_PATH)
    
    # Check that the result contains the timestamp in the format "YYYY-MM-DD_HH-MM-SS"
    expected_output = f"{model_train_mocked_timestamp.FEATURES_STORAGE_PATH}2000-01-01_00-00-00/"
    
    assert actual_output ==  expected_output
    
def test_feature_concat(common_objs_wrapper, sample_dataframe_1, exp_output_df_1):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper
    
    print(type(model_train_mocked_timestamp))
    actual_outcome = model_train_mocked_timestamp.run_feature_concat(sample_dataframe_1)
    
    assert 'CONCAT_FEATURE' in actual_outcome.schema.names
    assert 'CODE' in actual_outcome.schema.names

def test_handel_imbalanced_labels(common_objs_wrapper, sample_dataframe_1, exp_output_df_1, exp_output_df_2):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper

    results_dict = {
                exp_output_df_1: "replace", 
                exp_output_df_2: "filter", 
                sample_dataframe_1: None
               }        

    for exp_output, method in results_dict.items():

        model_train_mocked_timestamp.cfg.feature_transformation_cfg.handeling_imbalanced_label_instructions["how"]= method
        model_train_mocked_timestamp.cfg.feature_transformation_cfg.handeling_imbalanced_label_instructions["replacement_val"]= "99999999"
        model_train_mocked_timestamp.cfg.feature_transformation_cfg.handeling_imbalanced_label_instructions["threshold"]= 2
        actual_output = model_train_mocked_timestamp.handel_imbalanced_labels(sample_dataframe_1)

        assert spark_df_data_equal(actual_output, exp_output)
    
def test_spark_to_pandas_transformation_with_sampling(common_objs_wrapper, sample_dataframe_1):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper
    
    expectd_no_rows = 4
    model_train_mocked_timestamp.cfg.feature_transformation_cfg.sampling_instructions["n"]= expectd_no_rows
    actual_output = model_train_mocked_timestamp.run_spark_to_pandas_transformation_with_sampling(sample_dataframe_1)

    assert isinstance(actual_output, pd.DataFrame)
    assert len(actual_output)== expectd_no_rows
    
def test_spark_to_pandas_transformation_without_sampling(common_objs_wrapper, sample_dataframe_1):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper

    model_train_mocked_timestamp.cfg.feature_transformation_cfg.sampling_instructions= None
    
    actual_output = model_train_mocked_timestamp.run_spark_to_pandas_transformation_with_sampling(sample_dataframe_1)
    
    assert isinstance(actual_output, pd.DataFrame)
    assert sample_dataframe_1.count()== len(actual_output)
        
def test_label_transformation_and_seperation(common_objs_wrapper, sample_dataframe_1, exp_labels):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper
    
    sample_dataframe_1_pd = sample_dataframe_1.toPandas()
    expected_output_df = sample_dataframe_1_pd.drop(["CODE"], axis=1) 
    actual_output_df, actual_labels = model_train_mocked_timestamp.run_label_transformation_and_seperation(sample_dataframe_1_pd)

    assert actual_output_df.equals(expected_output_df)
    np.testing.assert_array_equal(actual_labels, exp_labels)

def test_training_tokenization_and_padding(common_objs_wrapper, sample_dataframe_2, vocab_size, max_text_len, exp_output_3):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper
    
    model_train_mocked_timestamp.cfg.feature_transformation_cfg.tokenization_instructions["max_features"] = vocab_size
    model_train_mocked_timestamp.cfg.feature_transformation_cfg.padding_instructions["max_text_len"] = max_text_len
  
    actual_output = model_train_mocked_timestamp.run_training_tokenization_and_padding(sample_dataframe_2)

    np.testing.assert_array_equal(actual_output, exp_output_3)
 
def test_train_validation_split(common_objs_wrapper, sample_data_3, validation_size): 
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper
    
    # Get test data
    X, y = sample_data_3
   
    # Call the train_validation_split function
    X_train, y_train, X_val, y_val = model_train_mocked_timestamp.train_validation_split(X, y, validation_size)

    # Assert that the training set and validation set have the correct size
    assert X_train.shape == (4, 2)
    assert y_train.shape == (4,)
    assert X_val.shape == (1, 2)
    assert y_val.shape == (1,)

    # Assert that the split ratio is correct
    assert np.isclose(len(X_val) / len(X), validation_size)
    
def test_fit_model(common_objs_wrapper, vocab_size, max_text_len, num_classes):
    dbutils, base_artifact_path, model_train_mocked_timestamp = common_objs_wrapper

    # Generate mock training data
    num_samples_train = 20
    num_samples_val = 4
    
    # paramter of the input shape to the model
    input_length = max_text_len
    input_dim = vocab_size
    
    # Generate mock train data
    X_train = np.random.randint(1, input_dim, size=(num_samples_train, input_length))
    y_train = np.random.randint(0, num_classes, size=num_samples_train)

    X_train = pad_sequences(X_train, maxlen=input_length)
    y_train = to_categorical(y_train, num_classes=num_classes)

    # Generate mock validation data
    X_val = np.random.randint(1, input_dim, size=(num_samples_val, input_length))
    y_val = np.random.randint(0, num_classes, size=num_samples_val)

    X_val = pad_sequences(X_val, maxlen=input_length)
    y_val = to_categorical(y_val, num_classes=num_classes)

    model_train_mocked_timestamp.cfg.train_instructions['input_dim'] = input_dim
    model_train_mocked_timestamp.cfg.train_instructions['input_length']= input_length
    model_train_mocked_timestamp.cfg.train_instructions["num_classes"] = num_classes

    
    model, history = model_train_mocked_timestamp.fit_model(X_train, y_train, X_val, y_val)  
    assert isinstance(model, tf.keras.Model)
    assert isinstance(history, tf.keras.callbacks.History)
    