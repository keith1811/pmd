"""
Tests for evaluating the model inference pipeline
"""
from modules.model_inference_module import ModelInference
from modules.utils.common import ModelInferenceConfig, FeatureTransformationConfig, IntermediateTransformationConfig, MLflowTrackingConfig
from modules.utils.notebook_utils import load_and_set_env_vars, load_config
from pyspark.sql import Row, SparkSession, DataFrame
from unittest.mock import patch

import mlflow
import pandas as pd
import pytest

data = [
    {
        "ID": "1",
        "SUPPLIER_ORG": "123",
        "BUYER_ORG": "456",
        "PO_ITEM_ID": "567329",
        "SUPPLIER_PART": "11111111",
        "MANUFACTURER_PART": "456123",
        "IS_ADHOC": "True",
        "MANUFACTURER_NAME": "Mercedes Infotainment",
        "DESCRIPTION": "In car stereo speakers",
        "ITEM_TYPE": "",
        "CODE": 15526645,
    },
    {
        "ID": "2",
        "SUPPLIER_ORG": "123",
        "BUYER_ORG": "456",
        "PO_ITEM_ID": "789456",
        "SUPPLIER_PART": "22222222",
        "MANUFACTURER_PART": "456123",
        "IS_ADHOC": "False",
        "MANUFACTURER_NAME": "BMW Interiors",
        "DESCRIPTION": "Leather seats",
        "ITEM_TYPE": "",
        "CODE": 56452749,
    },
]

def setup_function():

    # Set pipeline name
    pipeline_name = 'model_inference'

    # Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
    pipeline_config = load_config(pipeline_name, 'notebookdev')

    # Load and set arbitrary params via spark_env_vars
    # Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
    env_vars = load_and_set_env_vars(env='notebookdev')

    # Get model_uri registered by MLFlow
    model_name = pipeline_config['mlflow_params']['model_name']
    model_registry_stage = pipeline_config['mlflow_params']['model_registry_stage']

    model_uri = f'models:/{model_name}/{model_registry_stage}'
    print(f'model_uri: {model_uri}')

    # Set input path and table type as well as output table path
    inference_input_table_path = env_vars['inference_input_table_path']
    inference_input_table_type = env_vars['inference_input_table_type']

    batch_scoring = env_vars['batch_scoring']
    batch_size = env_vars['batch_size']
    save_output = env_vars['save_output']
    inference_output_table_path = env_vars['inference_output_table_path']

    # Set intermediate transformation config
    intermediate_transformation_cfg = IntermediateTransformationConfig(**pipeline_config['data_prep_params'])

    # Set mlflow config
    mlflow_tracking_cfg = MLflowTrackingConfig(model_name=pipeline_config['mlflow_params']['model_name'],
                                            registry_uri=mlflow.get_registry_uri(),
                                            model_registry_stage=pipeline_config['mlflow_params']['model_registry_stage'],
                                            model_version=1)


    # Set feature transformation config
    feature_transformation_cfg = FeatureTransformationConfig(**pipeline_config['feature_transformation'])

    cfg = ModelInferenceConfig(
        mlflow_tracking_cfg=mlflow_tracking_cfg,
        inference_input_table_path=inference_input_table_path,
        inference_input_table_type=inference_input_table_type,
        intermediate_transformation_cfg = intermediate_transformation_cfg,
        feature_transformation_cfg=feature_transformation_cfg,
        batch_scoring=batch_scoring,
        batch_size=int(batch_size),
        save_output=save_output,
        inference_output_table_path=inference_output_table_path,
        env='notebookdev'
    )
    global infer
    infer = ModelInference(cfg)


def test_spark_to_pandas_conversion(spark):
    # Assemble
    test_df = spark.createDataFrame(map(lambda x: Row(**x), data))
    expected_df = pd.DataFrame.from_dict(data)

    # Act
    transformed_df = infer.transform_spark_to_pandas(test_df)

    # Assert returned dataframe is a pandas dataframe
    assert isinstance(transformed_df, pd.DataFrame)
    pd.testing.assert_frame_equal(transformed_df, expected_df)

def test_run_raw_data_prep(spark):
    # Assemble
    test_df = spark.createDataFrame(map(lambda x: Row(**x), data))
    test_df_pd = pd.DataFrame.from_dict(data)

    # Act
    transformed_df = infer.run_raw_data_prep(test_df)
    with pytest.raises(TypeError) as exc_info:
        transformed_df = infer.run_raw_data_prep(test_df_pd)

    # Assert returned dataframe is a pyspark dataframe
    assert isinstance(transformed_df, DataFrame)
    assert "Expected input_df to be a" in str(exc_info.value)

def test_feature_concat(spark):
    # Assemble
    test_df = spark.createDataFrame(map(lambda x: Row(**x), data))
    output_text = ['123', '123']
    
    # Act
    processed_df = infer.run_feature_concat(test_df)
    
    # Assert
    assert 'CONCAT_FEATURE' in processed_df.schema.names
    # assert set(output_text) == set(processed_df.select('text').collect())

def test_join_predict_to_original_df(spark):
    # Assemble
    test_df = spark.createDataFrame(map(lambda x: Row(**x), data))
    pred_list = [1, 0]
    prob_list = [0.9, 0.3]
    
    # Act
    processed_df = infer.join_predict_to_original_df(test_df, pred_list, prob_list)
    
    # Assert
    assert 'predicted_label' in processed_df.schema.names
    assert 'predicted_label_proba' in processed_df.schema.names  