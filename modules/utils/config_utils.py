from modules.utils.notebook_utils import load_config


def get_model_stage(env):
    pipeline_config = load_config(pipeline_name='model_inference', env=env)
    model_registry_stage = pipeline_config['mlflow_params']['model_registry_stage']
    return model_registry_stage


def get_model_version(env):
    pipeline_config = load_config(pipeline_name='model_inference', env=env)
    model_version = pipeline_config['mlflow_params']['model_version']
    return model_version


def get_model_name(env):
    pipeline_config = load_config(pipeline_name='model_inference', env=env)
    model_name = pipeline_config['mlflow_params']['model_name']
    return model_name

def get_feature_concat_cols(env, pipeline_name='model_inference'):
    pipeline_config = load_config(pipeline_name=pipeline_name, env=env)
    col_name_list = pipeline_config['feature_transformation']['feature_concat_instructions']['concat_cols']
    return col_name_list

def get_feature_concat_name(env, pipeline_name='model_inference'):
    pipeline_config = load_config(pipeline_name=pipeline_name, env=env)
    feature_concat_name = pipeline_config['feature_transformation']['text_col_name']
    return feature_concat_name

def get_replacement_value(env):
    pipeline_config = load_config(pipeline_name='model_inference', env=env)
    replacement_value = pipeline_config['data_prep_params']['handle_str_na_vals']['replacement_val']
    return replacement_value

def get_na_value_list(env):
    pipeline_config = load_config(pipeline_name='model_inference', env=env)
    va_value_list = pipeline_config['data_prep_params']['handle_str_na_vals']['na_vals_list']
    return va_value_list