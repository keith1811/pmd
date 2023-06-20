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

from modules.utils.logger_utils import get_logger
from modules.model_inference_module import ModelInference
from modules.utils.common import ModelInferenceConfig, FeatureTransformationConfig, IntermediateTransformationConfig, MLflowTrackingConfig
from modules.utils.notebook_utils import load_and_set_env_vars, load_config

_logger = get_logger()

# COMMAND ----------

# Set pipeline name
pipeline_name = 'model_inference'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name, dbutils.widgets.get('env'))

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# Set input path and table type as well as output table path
inference_input_table_path = env_vars['inference_input_table_path']
inference_input_table_type = env_vars['inference_input_table_type']

batch_scoring = env_vars['batch_scoring']
batch_size = env_vars['batch_size']
save_output = env_vars['save_output']
inference_output_table_path = env_vars['inference_output_table_path']

# COMMAND ----------

# Set intermediate transformation config
intermediate_transformation_cfg = IntermediateTransformationConfig(**pipeline_config['data_prep_params'])

# Set mlflow config
mlflow_tracking_cfg = MLflowTrackingConfig(model_name=pipeline_config['mlflow_params']['model_name'],
                                           registry_uri=env_vars["registry_uri"],
                                           model_registry_stage=pipeline_config['mlflow_params']['model_registry_stage'],
                                           model_version=pipeline_config['mlflow_params']['model_version'])


# Set feature transformation config
feature_transformation_cfg = FeatureTransformationConfig(**pipeline_config['feature_transformation'])

cfg = ModelInferenceConfig(mlflow_tracking_cfg=mlflow_tracking_cfg,
                            inference_input_table_path=inference_input_table_path,
                            inference_input_table_type=inference_input_table_type,
                            intermediate_transformation_cfg = intermediate_transformation_cfg,
                            feature_transformation_cfg=feature_transformation_cfg,
                            batch_scoring=batch_scoring,
                            batch_size=int(batch_size),
                            save_output=save_output,
                            inference_output_table_path=inference_output_table_path,
                            env=dbutils.widgets.get('env'))


# COMMAND ----------

model_inference_pipeline = ModelInference(cfg)
model_inference_pipeline.run()

