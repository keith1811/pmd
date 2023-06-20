import os
import pytest
import yaml
from decimal import Decimal

from delta import DeltaTable
from pyspark.sql.types import *
from pyspark.sql import Row
from modules.model_train_feature_creator_module import TrainFeatureCreator
from modules.model_inference_module import ModelInference
from modules.utils.common import *
from modules.utils.notebook_utils import load_and_set_env_vars, load_config
from modules.model_train_module import ModelTrain
from mock import patch, Mock


@pytest.fixture(scope="module")
def create_train_data(spark, dbutils, current_test_dir, base_testdata_path):
    dbutils.fs.rm(os.path.normpath(current_test_dir + f"/{base_testdata_path}/"), True)

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

    data = [(None, "12345", "1", "KRONOS Enterprise Archive V8", "22A", "MAN1", "001", "Material", "1", "12345678"),
            ("88888", "34567", "Nan", "CHARGING IN PROGRESS SIGN", "22A", "MAN1", "Null", "Material", "1", "23450189Z"),
            ("88888", "34567", "7", "CHARGING IN PROGRESS SIGN", "22A", "MAN1", "Null", "Material", "0", "23450189"),
            ("77777", "45678", "2", "Pump panel sign", "22A", "MAN1", "?", "Material", "1", "12345678"),
            ("Not Available", "56789", "3", "Mobile Case", "22A", "MAN1", "002", "Material", "0", "12345678"),
            ("66666", "56789", "4", "Mobile Case", "22A", "MAN1", "002", "Material", "0", "23450189"),
            ("Not Available", "56789", "0", "Mobile Case", "22A", "MAN1", "002", "Material", "0", "78965432"),
            ("77777", "43235", "5", "Content: Globe Prepaid Phone Stand", "22A", "MAN1", "nan", "Material", "1",
             "56789101"),
            ("77777", "45678", "6", "Pump panel sign", "22A", "MAN1", "?", "Material", "1", None),
            ]
    spark.createDataFrame(data, schema).coalesce(1).write.format("csv").mode('overwrite').option("header", "true").save(
        os.path.normpath(current_test_dir + f"/{base_testdata_path}/sample_dataframe"))

    """Mock sample reference df"""
    schema = StructType(
        [
            StructField("CODE", StringType(), True),
        ]
    )

    data = [("12345678",),
            ("23450189",),
            ("78965432",),
            ("56789101",),
            ("45678932",),
            ("Z4567891",),
            ]

    spark.createDataFrame(data, schema).coalesce(1).write.format("csv").mode('overwrite').option("header", "true").save(
        os.path.normpath(current_test_dir + f"/{base_testdata_path}/sample_ref_df"))


@pytest.fixture(scope="module")
def create_inference_data(spark, current_test_dir, base_testdata_path):
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
            "SBN_PREDICATED_UNSPSC_SEGMENT": "10",
            "SBN_PREDICATED_UNSPSC_FAMILY": "10",
            "SBN_PREDICATED_UNSPSC_CLASS": "10",
            "SBN_PREDICATED_UNSPSC_COMMODITY": "10",
            "SBN_PREDICTION_CONFIDENCE_SEGMENT": 0.78910,
            "SBN_PREDICTION_CONFIDENCE_FAMILY": 0.78910,
            "SBN_PREDICTION_CONFIDENCE_CLASS": 0.78910,
            "SBN_PREDICTION_CONFIDENCE_COMMODITY": 0.78910,
            "SBN_PREDICTION_LASTUPDATED_AT": "",
            "_DELTA_UPDATED_ON": "",
            "_DELTA_CREATED_ON": "",
            "CONCAT_FEATURE": "",
            "MODEL_UUID": "mid-01",
            "UUID": "mid-01",
            "NAME": "po_classification",
            "VERSION": 1,
            "STAGE": "Staging"
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
            "SBN_PREDICATED_UNSPSC_SEGMENT": "10",
            "SBN_PREDICATED_UNSPSC_FAMILY": "10",
            "SBN_PREDICATED_UNSPSC_CLASS": "10",
            "SBN_PREDICATED_UNSPSC_COMMODITY": "10",
            "SBN_PREDICTION_CONFIDENCE_SEGMENT": 0.78910,
            "SBN_PREDICTION_CONFIDENCE_FAMILY": 0.78910,
            "SBN_PREDICTION_CONFIDENCE_CLASS": 0.78910,
            "SBN_PREDICTION_CONFIDENCE_COMMODITY": 0.78910,
            "SBN_PREDICTION_LASTUPDATED_AT": "",
            "_DELTA_UPDATED_ON": "",
            "_DELTA_CREATED_ON": "",
            "CONCAT_FEATURE": "",
            "MODEL_UUID": "mid-01",
            "UUID": "mid-01",
            "NAME": "model_name",
            "VERSION": 2,
            "STAGE": "production"
        },
    ]
    spark.createDataFrame(map(lambda x: Row(**x), data)).coalesce(1).write.format("delta").mode('overwrite').option(
        "header", "true").save(os.path.normpath(current_test_dir + f"/{base_testdata_path}/inference_input_table"))

    """Mock inference source df."""
    schema = StructType(
        [
            StructField("PO_ITEM_ID", StringType(), True),
            StructField("CONCAT_FEATURE", StringType(), True),
            StructField("predicted_label", StringType(), True),
            StructField("SBN_PREDICATED_UNSPSC_SEGMENT", StringType(), True),
            StructField("SBN_PREDICATED_UNSPSC_FAMILY", StringType(), True),
            StructField("SBN_PREDICATED_UNSPSC_CLASS", StringType(), True),
            StructField("SBN_PREDICATED_UNSPSC_COMMODITY", StringType(), True),
            StructField("predicted_label_proba", DecimalType(6, 5), True),
            StructField("SBN_PREDICTION_CONFIDENCE_SEGMENT", DecimalType(6, 5), True),
            StructField("SBN_PREDICTION_CONFIDENCE_FAMILY", DecimalType(6, 5), True),
            StructField("SBN_PREDICTION_CONFIDENCE_CLASS", DecimalType(6, 5), True),
            StructField("SBN_PREDICTION_CONFIDENCE_COMMODITY", DecimalType(6, 5), True),
            StructField("SBN_PREDICTION_LASTUPDATED_AT", TimestampType(), True),
            StructField("MODEL_UUID", StringType(), True),
            StructField("_DELTA_CREATED_ON", TimestampType(), True),
            StructField("_DELTA_UPDATED_ON", TimestampType(), True),
            StructField("UUID", StringType(), True),
            StructField("NAME", StringType(), True),
            StructField("VERSION", IntegerType(), True),
            StructField("STAGE", StringType(), True)
        ]
    )
    data = [(123, "test description", *("111111" for _ in range(5)), *(Decimal(0.78900) for _ in range(5)), None,
             'MID-001', None, None, "MID-001", "po_classification", 1, "Staging")]

    spark.createDataFrame(data, schema).coalesce(1).write.format("delta").mode('overwrite').option("header",
                                                                                                   "true").save(
        os.path.normpath(current_test_dir + f"/{base_testdata_path}/model_info_df"))


@pytest.fixture(scope="module")
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
    cfg = ModelTrainConfig(mlflow_tracking_cfg=mlflow_tracking_cfg,
                           intermediate_table_cfg=intermediate_table_cfg,
                           feature_transformation_cfg=feature_transformation_cfg,
                           train_val_split_config=pipeline_config["train_val_split_config"],
                           train_instructions=pipeline_config["train_instructions"],
                           pipeline_config=pipeline_config)


@pytest.fixture(scope="module")
def model_train_mocked_timestamp(setup_train, monkeymodule, base_artifact_path):
    # Use monkeymodule to replace get_current_timestamp with a mock object
    timestamp_mock_method = Mock(return_value="2000-01-01_00-00-01")
    monkeymodule.setattr("modules.model_train_module.get_current_timestamp", timestamp_mock_method)

    # Use monkeymodule to set the value of BASE_ARTIFACT_PATH to base_artifact_path
    monkeymodule.setattr("modules.model_train_module.ModelTrain.BASE_ARTIFACT_PATH", base_artifact_path)
    # Use monkeymodule to set the value of BASE_ARTIFACT_PATH to base_artifact_path
    monkeymodule.setattr("modules.model_train_module.ModelTrain.FEATURES_STORAGE_PATH", base_artifact_path)

    # Initiate train class
    model_train = ModelTrain(cfg)
    yield model_train


@pytest.fixture(scope="module")
def setup_inference():
    # Set pipeline name
    pipeline_name = 'model_inference'

    # Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
    pipeline_config = load_config(pipeline_name, 'ml')

    # Load and set arbitrary params via spark_env_vars
    # Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
    env_vars = load_and_set_env_vars(env='ml')

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
                                               model_registry_stage=pipeline_config['mlflow_params'][
                                                   'model_registry_stage'],
                                               model_version=1)

    # Set feature transformation config
    feature_transformation_cfg = FeatureTransformationConfig(**pipeline_config['feature_transformation'])

    cfg = ModelInferenceConfig(
        mlflow_tracking_cfg=mlflow_tracking_cfg,
        inference_input_table_path=inference_input_table_path,
        inference_input_table_type=inference_input_table_type,
        intermediate_transformation_cfg=intermediate_transformation_cfg,
        feature_transformation_cfg=feature_transformation_cfg,
        batch_scoring=batch_scoring,
        batch_size=int(batch_size),
        save_output=save_output,
        inference_output_table_path=inference_output_table_path,
        env='ml'
    )
    global infer
    infer = ModelInference(cfg)


@pytest.fixture
def get_table_storage_location_mocker(current_test_dir, base_testdata_path):
    with patch("utils.sbnutils.get_table_storage_location") as table_storage_location:
        table_storage_location.return_value = os.path.normpath(
            current_test_dir + f"/{base_testdata_path}/model_info_df")
        yield table_storage_location


@pytest.fixture
def get_delta_table_mocker(spark, current_test_dir, base_testdata_path):
    with patch("utils.sbnutils.get_delta_table") as get_delta_table:
        table_location = current_test_dir + f"/{base_testdata_path}/inference_input_table"
        get_delta_table.return_value = DeltaTable.forPath(spark, table_location)
        yield get_delta_table


##############################################################################################
## Full end-to-end unit tests of featurization/training/inference for ml use cases
##############################################################################################

def test_run_feature_table_creator(create_train_data, current_test_dir, base_testdata_path):
    # Set pipeline name
    pipeline_name = 'model_train_feature_creator'

    # Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
    pipeline_config = load_config(pipeline_name, 'ml')

    # Set FeaturizerConfig - data preparation config
    intermediate_transformation_cfg = IntermediateTransformationConfig(**pipeline_config['data_prep_params'])

    # Set Intermediate table config
    intermediate_table_cfg = IntermediateTableConfig(
        storage_path=os.path.normpath(current_test_dir + f"/{base_testdata_path}/sample_dataframe_1"))
    # Set Feature table config

    # Set FeatureTableCreatorConfig
    cfg = FeatureTableCreatorConfig(input_tables=yaml.safe_load(f"""
                                                                production_2022_community_0:
                                                                    path: {os.path.normpath(current_test_dir + '/' + base_testdata_path + '/sample_dataframe')}
                                                                unspsc_reference_table:
                                                                    path: {os.path.normpath(current_test_dir + '/' + base_testdata_path + '/sample_ref_df')}
                                                                """),
                                    intermediate_transformation_cfg=intermediate_transformation_cfg,
                                    intermediate_table_cfg=intermediate_table_cfg)

    # Instantiate pipeline
    feature_table_creator_pipeline = TrainFeatureCreator(cfg)
    feature_table_creator_pipeline.run()


def test_run_train(model_train_mocked_timestamp, current_test_dir, base_testdata_path):
    model_train_mocked_timestamp.cfg.intermediate_table_cfg.storage_path = os.path.normpath(
        current_test_dir + f"/{base_testdata_path}/sample_dataframe_1")
    model_train_mocked_timestamp.cfg.feature_transformation_cfg.handeling_imbalanced_label_instructions["threshold"] = 1
    model_train_mocked_timestamp.cfg.feature_transformation_cfg.sampling_instructions["n"] = 800
    model_train_mocked_timestamp.cfg.feature_transformation_cfg.sampling_instructions["replace"] = True
    model_train_mocked_timestamp.run()


def test_load_model(setup_inference):
    # Assemble

    # Act
    model = infer.load_model()

    # Assert
    assert isinstance(model, mlflow.pyfunc.PyFuncModel)


def test_run_inference(dbutils, setup_inference, create_inference_data, current_test_dir, base_testdata_path,
                       get_table_storage_location_mocker, get_delta_table_mocker):
    infer.cfg.inference_input_table_path = os.path.normpath(
        current_test_dir + f"/{base_testdata_path}/inference_input_table")
    infer.cfg.inference_input_table_type = "delta"
    infer.cfg.mlflow_tracking_cfg.model_version = 1
    infer.cfg.batch_scoring = "True"
    infer.cfg.inference_output_table_path = os.path.normpath(
        current_test_dir + f"/{base_testdata_path}/inference_output_table")
    infer.run()

    infer.cfg.batch_scoring = "False"
    infer.run()
    dbutils.fs.rm(os.path.normpath(current_test_dir + f"/{base_testdata_path}/"), True)