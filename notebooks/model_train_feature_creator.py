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

from modules.model_train_feature_creator_module import TrainFeatureCreator
from modules.utils.common import IntermediateTransformationConfig, IntermediateTableConfig, FeatureTableCreatorConfig

_logger = get_logger()

# COMMAND ----------

# Set pipeline name
pipeline_name = 'model_train_feature_creator'

# Load pipeline config yaml file (../conf/pipeline_configs/{pipeline_name}.yml)
pipeline_config = load_config(pipeline_name)

# Load and set arbitrary params via spark_env_vars
# Params passed via ../conf/{env}/.{env}.env and ../conf/.base_data_params
env_vars = load_and_set_env_vars(env=dbutils.widgets.get('env'))

# COMMAND ----------

# Set FeaturizerConfig - data preparation config
intermediate_transformation_cfg = IntermediateTransformationConfig(**pipeline_config['data_prep_params'])

# Set Intermediate table config
intermediate_table_cfg = IntermediateTableConfig(storage_path=env_vars['intermediate_table_storage_path'])
# Set Feature table config

# Set FeatureTableCreatorConfig
cfg = FeatureTableCreatorConfig(input_tables=pipeline_config['raw_input_tables'],
                                intermediate_transformation_cfg=intermediate_transformation_cfg,
                                intermediate_table_cfg=intermediate_table_cfg)

# COMMAND ----------

sql("SET spark.databricks.delta.formatCheck.enabled=false")

# COMMAND ----------

# Instantiate pipeline
feature_table_creator_pipeline = TrainFeatureCreator(cfg)
feature_table_creator_pipeline.run()
