from typing import Dict, Any

import mlflow
from dataclasses import dataclass


@dataclass
class IntermediateTransformationConfig:
    """
     Attributes:
        handle_str_na_vals: dict

        select_cols_instructions: dict

        drop_duplicates_instructions: dict

        handle_missing_vals: dict
            Config to indicate whether or not to drop missing values and how to handle them
    """
    unspsc_processsing: dict = None
    handle_str_na_vals: dict = None
    select_cols_instructions: dict = None
    drop_duplicates_instructions: dict = None
    handle_missing_vals: dict = None


@dataclass
class IntermediateTableConfig:
    """
    Configuration data class used to unpack parameters when creating or loading intermediate table.
    Attributes:
        database_name str
            Name of database to use for creating the intermediate table
        table_name: str
            Name of processed intermediate table
        storage_path: str
            [Optional] string containing path to intermediate table storage path
    """
    database_name: str = None
    table_name: str = None
    storage_path: str = None


@dataclass
class FeatureTransformationConfig:
    """
     Attributes:
            label_col: str
            text_col_name: str
            feature_concat_instructions: dict
            handeling_imbalanced_label_instructions: dict
            sampling_instructions: dict
            tokenization_instructions: dict
            padding_instructions: dict
    """
    label_col: str = None
    text_col_name: str = None
    feature_concat_instructions: dict = None
    handeling_imbalanced_label_instructions: dict = None
    sampling_instructions: dict = None
    tokenization_instructions: dict = None
    padding_instructions: dict = None


@dataclass
class FeatureTableConfig:
    """
    Configuration data class used to unpack parameters when creating or loading feature table.
    Attributes:
        le_storage_path: str
            string containing path to fitted label encoder storage directory
        tfidf_vec_storage_path: str
            string containing path to fitted tfidf vectorizer storage directory
        features_storage_path: str
            string containing path to feature table storage directory
    """
    function_and_features_storage_path: str = None


@dataclass
class FeatureTableCreatorConfig:
    """
    Attributes:
        input_table: str
            Name of the table to use as input for creating features
        run_stages: list
            which stages of transformation to run
        intermediate_transformation_cfg: IntermediateTransformationConfig
            Intermediate data transformation config to specify filter, drop duplicates and handle missing values params
        intermediate_table_cfg: IntermediateTableConfig
            Intermediate table config to specify database_name, table_name and storage path
    """
    input_tables: dict
    intermediate_transformation_cfg: IntermediateTransformationConfig = None
    intermediate_table_cfg: IntermediateTableConfig = None


@dataclass
class MLflowTrackingConfig:
    """
    Configuration data class used to unpack MLflow parameters during model training/inference run.

    Attributes:
        run_name: str
            Name of MLflow run
        experiment_id: int
            ID of the MLflow experiment to be activated. If an experiment with this ID does not exist, raise an exception.
        experiment_path: str
            Case sensitive name of the experiment to be activated. If an experiment with this name does not exist,
            a new experiment wth this name is created.
        model_name: str
            Name of the registered model under which to create a new model version. If a registered model with the given
            name does not exist, it will be created automatically.
        registry_uri: str
            The registry uri for sharing the model across all workspaces
        model_registry_stage: str
            The registry stage to be used for loading the model from databricks model registry
        model_version: int
            Version of MLflow model
    """
    run_name: str = None
    experiment_id: int = None
    experiment_path: str = None
    model_name: str = None 
    registry_uri: str = None
    model_registry_stage: str = None 
    model_version: int = None
     
@dataclass
class ModelTrainConfig:
    """
    Configuration model train class

    Attributes:
        mlflow_tracking_cfg: MLflowTrackingConfig
        intermediate_table_cfg: IntermediateTableConfig
        feature_transformation_cfg: FeatureTransformationConfig
        train_val_split_config: Dict[str, Any]
        train_instructions: Dict[str, Any]

        pipeline_config: Dict[str, Any]

    """
    mlflow_tracking_cfg: MLflowTrackingConfig
    intermediate_table_cfg: IntermediateTableConfig
    feature_transformation_cfg: FeatureTransformationConfig
    train_val_split_config: Dict[str, Any]
    train_instructions: Dict[str, Any]
    pipeline_config: Dict[str, Any] = None


@dataclass
class ModelInferenceConfig:
    """
    Configuration model inference class

    Attributes:
        mlflow_tracking_cfg: MLflowTrackingConfig
        
        inference_input_table_path: str

        inference_input_table_type: str

        feature_transformation_cfg: FeatureTransformationConfig

        batch_scoring: str

        save_output: str

        inference_output_table_path: str

        env: str
    """
    mlflow_tracking_cfg: MLflowTrackingConfig
    inference_input_table_path: str
    inference_input_table_type: str
    intermediate_transformation_cfg: IntermediateTransformationConfig
    feature_transformation_cfg: FeatureTransformationConfig
    batch_scoring: str
    batch_size: int
    save_output: str
    inference_output_table_path: str
    env: str

class KerasModel(mlflow.pyfunc.PythonModel):
    """
    A custom MLflow PythonModel for a Keras deep learning model that predicts classes for input data.
    
    Methods
    -------
    predict(context, input_data)
        Make predictions using the trained Keras model and return them as a NumPy array.
    
    _load_tokenizer()
        Load the saved tokenizer object from disk and return it.
        
    _load_label_encoder()
        Load the saved label encoder object from disk and return it.
    """
    def __init__(self, model, tokenizer, label_encoder):
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
    
    def _load_tokenizer(self):
        return self.tokenizer
    
    def _load_label_encoder(self):
        return self.label_encoder

    def predict(self, context, input_data):
        # Make predictions using the trained Keras model
        y_pred = self.model.predict(input_data)
    
        # Return the predictions as a Pandas dataframe
        return y_pred    