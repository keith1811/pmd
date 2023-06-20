# Databricks notebook source
!cp ../requirements-ml.txt ~/.
%pip install -r ~/requirements-ml.txt

# COMMAND ----------

dbutils.widgets.dropdown('env', 'ml', ['ml'], 'Environment Name')

# COMMAND ----------

# once you updated the modules within ./src folder you need to run this
%reload_ext autoreload
%autoreload 2

# COMMAND ----------

from modules.utils.notebook_utils import load_and_set_env_vars, load_config
from modules.utils.logger_utils import get_logger

from modules.model_train_module import ModelTrain
from modules.utils.common import ModelTrainConfig, MLflowTrackingConfig, IntermediateTableConfig, FeatureTransformationConfig

_logger = get_logger()

# COMMAND ----------

# Set pipeline name
pipeline_name = 'model_train'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name, dbutils.widgets.get('env'))

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# Set MLflowTrackingFlow
mlflow_tracking_cfg = MLflowTrackingConfig(run_name=pipeline_config["mlflow_params"]["run_name"],
                                           model_name=pipeline_config["mlflow_params"]["model_name"],
                                           registry_uri=env_vars["registry_uri"])

# Get Intermediate storage path
intermediate_table_cfg = IntermediateTableConfig(storage_path=env_vars["intermediate_table_storage_path"])

# Set feature transformation config
feature_transformation_cfg = FeatureTransformationConfig(**pipeline_config["feature_transformation"])

# Set ModelTrainConfig
cfg = ModelTrainConfig(mlflow_tracking_cfg = mlflow_tracking_cfg,
                       intermediate_table_cfg = intermediate_table_cfg,
                       feature_transformation_cfg = feature_transformation_cfg,
                       train_val_split_config = pipeline_config["train_val_split_config"],
                       train_instructions = pipeline_config["train_instructions"],
                       pipeline_config = pipeline_config)

# COMMAND ----------

# Instantiate pipeline
model_train_pipeline = ModelTrain(cfg)
model_train_pipeline.run()

# COMMAND ----------


