from typing import Tuple

import mlflow
import mlflow.keras
import mlflow.tensorflow

import joblib
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils import to_categorical

import pyspark.sql.dataframe
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support

from modules.utils.logger_utils import get_logger
from modules.utils.train_and_evaluation import ModelTrainer, log_acc_and_loss_plot
from modules.utils.common import MLflowTrackingConfig, ModelTrainConfig, KerasModel
from modules.utils.feature_transform_utils import prepare_corpus_for_model, pad_corpus
from modules.utils.spark_utils import replace_infrequent_values, filter_by_unique_values_count
from modules.utils.persist_and_load_utils import load_table, persist_to_numpy, persist_pandas_to_csv

def get_current_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

class ModelTrain:

    # Fix base path for storing artifact during training
    BASE_ARTIFACT_PATH = "dbfs:/tmp/training/"
    FEATURES_STORAGE_PATH = "/dbfs/tmp/training/"

    """
    Class for execute model training
    """
    def __init__(self, cfg: ModelTrainConfig):
        global _logger
        global spark
        global dbutils
        _logger = get_logger()
        spark = SparkSession.builder.getOrCreate()
        dbutils = DBUtils(spark)
        self.cfg = cfg
        self.dbfs_local_storage_path = self._create_unique_dbfs_storage_dir(dbutils, self.BASE_ARTIFACT_PATH)
    
    @staticmethod
    def _create_unique_dbfs_storage_dir(dbutils, base_path: str = "dbfs:/tmp/training/") -> str:
        """
        Create a unique dbfs path to store features and artifacts

        Parameters
        ----------
        base_path: str
            base dbfs path

        Returns
        ----------
        str: The dbfs path where the features and artifact are 
        going to be persisted.
        """
        # get current timestamp
        timestamp = get_current_timestamp()
        
        unique_storage_path = base_path + timestamp + "/"
        # Remove any existing files in the unique_storage_path
        dbutils.fs.rm(unique_storage_path, True)

        # Create the unique_storage_path
        dbutils.fs.mkdirs(unique_storage_path)

        # Persist features to the given dbfs path from env variable
        features_storage_path = ModelTrain.FEATURES_STORAGE_PATH + timestamp + "/"

        return features_storage_path

    @staticmethod
    def _set_experiment(mlflow_tracking_cfg: MLflowTrackingConfig):
        """
        Set MLflow experiment. Use one of either experiment_id or experiment_path
        """
        
        if mlflow_tracking_cfg.experiment_id is not None:
            _logger.info(f'MLflow experiment_id: {mlflow_tracking_cfg.experiment_id}')
            mlflow.set_experiment(experiment_id=mlflow_tracking_cfg.experiment_id)
        elif mlflow_tracking_cfg.experiment_path is not None:
            _logger.info(f'MLflow experiment_path: {mlflow_tracking_cfg.experiment_path}')
            mlflow.set_experiment(experiment_name=mlflow_tracking_cfg.experiment_path)
        else:
            raise RuntimeError('MLflow experiment_id or experiment_path must be set in mlflow_params')
    
    def run_feature_concat(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame: 
        """
        Run features concatenation to one single column

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Given Spark DataFrame

        Returns
        -------
        pyspark.sql.DataFrame
            sub-sampled Spark DataFrame
        """ 
        label_col = self.cfg.feature_transformation_cfg.label_col
        feature_concat_instructions = self.cfg.feature_transformation_cfg.feature_concat_instructions
        
        df_concatenated = df.select(f.concat_ws(" ", *feature_concat_instructions["concat_cols"]).alias(feature_concat_instructions["concat_col_name"]), 
                                    label_col)  
        _logger.info(f'''Following columns {feature_concat_instructions["concat_cols"]} concatenated to single '{feature_concat_instructions["concat_col_name"]}' column''')
        
        return df_concatenated

    def handel_imbalanced_labels(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Handel infrequent unspsc codes based on the provided threshold with
        three functionality:
        
        1. Replace infrequent unspsc by given replacement value to omit minority classes 
           and have a well balanced dataset
        2. Drop the rows with minority classes
        3. Do nothing 
        
        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Given Spark DataFrame

        Returns
        -------
        pyspark.sql.DataFrame
            Updated Spark DataFrame
        """ 
        handeling_imbalanced_label_instructions = self.cfg.feature_transformation_cfg.handeling_imbalanced_label_instructions
        self.handel_imbalanced_labels_method = None
        self.infrequent_labels_pdf = None
        if handeling_imbalanced_label_instructions["how"]=="replace":
            df, infrequent_labels_sdf = replace_infrequent_values(df, 
                                                               handeling_imbalanced_label_instructions["col_name"], 
                                                               handeling_imbalanced_label_instructions["threshold"],
                                                               handeling_imbalanced_label_instructions["replacement_val"])
            self.infrequent_labels_pdf = infrequent_labels_sdf.toPandas()
            self.handel_imbalanced_labels_method= handeling_imbalanced_label_instructions["how"]
            
        elif handeling_imbalanced_label_instructions["how"]=="filter":
            df, self.infrequent_labels_pdf = filter_by_unique_values_count(df, 
                                                             handeling_imbalanced_label_instructions["col_name"], 
                                                             handeling_imbalanced_label_instructions["threshold"])
            self.handel_imbalanced_labels_method = handeling_imbalanced_label_instructions["how"]
            
        else:
            _logger.info(f'No action implemented for imbalanced label improvement')
        
        if self.infrequent_labels_pdf is not None:
            persist_pandas_to_csv(self.infrequent_labels_pdf, self.dbfs_local_storage_path + "infrequent_labels.csv")
            _logger.info(f'{len(self.infrequent_labels_pdf)} Infrequent labels are stored successfully within a table in: {self.dbfs_local_storage_path}')

        _logger.info(f'Dataframe has {df.count()} rows after label imbalanced handeling')    

        return df
    
    def run_spark_to_pandas_transformation_with_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run data sampling function to return a fraction of the given dataframe

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Given Spark DataFrame

        Returns
        -------
        pd.DataFrame
            sub-sampled Pandas DataFrame
        """
        sampling_instructions = self.cfg.feature_transformation_cfg.sampling_instructions
        
        # Transform pyspark to pandas dataframe
        pdf = df.toPandas()

        # Apply sampling
        if sampling_instructions:
            pdf = pdf.sample(**sampling_instructions)
            _logger.info(f'Dimension of sub-sampled table is: {pdf.shape}')  
        
        return pdf
    
    def run_label_transformation_and_seperation(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
        """
        Transform labels of the given pandas dataframe 
        1. Fit and transform label encoding from sklearn.preprocessing to transform labels to index (int)
        2. store the fitted label encoded to the given env variable
        3. separate label columns from dataframe
        4. transform the indexed labels into categorical format using keras.utils (one-hot-encoded)
       
        Parameters
        ----------
        df: pd.DataFrame
            Given Pandas DataFrame

        Returns
        -------
        X_train: pd.DataFrame
            Pandas DataFrame without labels
        y_train: np.ndarray
            A binary matrix representation of the input. The class axis is placed
            last.
        """
        label_col = self.cfg.feature_transformation_cfg.label_col
        
        # fit and transform label encoder
        self.label_encoder = LabelEncoder()
        df[label_col] = self.label_encoder.fit_transform(df[label_col])

        # separate label column from input dataframe
        df_without_label_col = df.drop(label_col, axis=1)
        y_train = df[label_col]
        
        # apply one-hot-encoding to the indexed labels
        y_train = to_categorical(y_train)
        _logger.info(f'Dimensions of labels after categorical transformation: {y_train.shape}') 
        
        # dump label encoder to the given directory
        joblib.dump(self.label_encoder, self.dbfs_local_storage_path + "label_encoder.pkl" )
        _logger.info(f'Fitted label encoder is stored in dbfs successfully in: {self.dbfs_local_storage_path + "label_encoder.pkl" }')
        
        return df_without_label_col, y_train
    
    def run_training_tokenization_and_padding(self, df: pd.DataFrame) -> np.ndarray: 
        """
        Paramerters
        -----------
            df: pd.DataFrame
                given training dataframe containing text column(without label column)
        
        Returns
        -------
            np.ndarray:
                An array containing one-hot-encoded corpus 
        """
        tokenization_instructions = self.cfg.feature_transformation_cfg.tokenization_instructions
        padding_instructions = self.cfg.feature_transformation_cfg.padding_instructions
        
        train_seq, self.tokenizer = prepare_corpus_for_model(df, 
                                                             tokenization_instructions.get("tokenized_col_name"), 
                                                             tokenization_instructions.get("max_features"))
        train_padded_corpus = pad_corpus(train_seq, padding_instructions.get("max_text_len"))
        
        _logger.info(f'Dimensions of corpus after one-hot-encoding and padding: {train_padded_corpus.shape}')
        
        # dump label encoder to the given directory
        joblib.dump(self.tokenizer, self.dbfs_local_storage_path + "tokenizer.pkl" )
        _logger.info(f'Fitted tokenizer is stored in dbfs successfully in: {self.dbfs_local_storage_path + "tokenizer.pkl"}')
        
        return train_padded_corpus
    
    def persist_features_to_dbfs(self, feature_arr: np.ndarray, label_arr: np.ndarray) -> None: 
        """
        This function creates a unique storage directory path with the current timestamp and saves training features and 
        labels to that directory path in numpy file format. The fitted label encoder is also persisted to the same 
        directory path in pickle file format.
    
        Parameters
        ----------
        feature_arr: np.ndarray
            A numpy ndarray representing the training features.
        label_arr: np.ndarray
            A numpy ndarray representing the training labels.
        """
        # persist features to the given dbfs path
        persist_to_numpy(feature_arr, self.dbfs_local_storage_path + "X_train.npy")
        persist_to_numpy(label_arr, self.dbfs_local_storage_path + "y_train.npy")
        _logger.info(f'Processsd training Features and label are stored successfully in: {self.dbfs_local_storage_path}')
    
    def run_feature_preparation_for_training(self)-> Tuple[np.ndarray, np.ndarray]:
        """
        Run feature preparation for training from intermediate
        """
        _logger.info('Initiated Feature transformation ....')
        _logger.info('==========Load Intermediate Table==========')
        intermediate_df = load_table(self.cfg.intermediate_table_cfg.storage_path, table_type=None, spark = spark)

        _logger.info('==========Concatenate Features To One Column==========')
        concatenated_features_df = self.run_feature_concat(intermediate_df) 
        
        _logger.info('==========Handeling Infrequent UNSPSC Values==========')
        df_with_balanced_labels = self.handel_imbalanced_labels(concatenated_features_df)
        
        _logger.info('==========Transform Spark To Pandas Datafram & Sample Data==========')
        sampled_df = self.run_spark_to_pandas_transformation_with_sampling(df_with_balanced_labels)

        _logger.info('==========Apply Label-Encoding & Label Separatation==========')
        X_train, y_train= self.run_label_transformation_and_seperation(sampled_df)

        _logger.info('==========Transform Train Text Column Using Tokenization & Padding==========')
        train_padded_corpus = self.run_training_tokenization_and_padding(X_train)
        
        _logger.info('==========Persist Pre-processed Trianing data==========')
        self.persist_features_to_dbfs(train_padded_corpus, y_train)  
        
        return train_padded_corpus, y_train
    
    @staticmethod
    def train_validation_split(X: np.ndarray, 
                               y: np.ndarray, 
                               val_size: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Split the data into random train and validation sets.

        Parameters
        -----------
            X: np.ndarray: 
                The data to be split.
            y: np.ndarray: 
                The labels to be split.
            val_size: float = 0.9
                Split size

        Returns
        -----------
            tuple: Returns a tuple of (X_train, y_train, X_val, y_val) where X_train and y_train are 
            the training data and labels, and X_val and y_val are the validation data and labels.
        """
        # Compute the number of samples in the validation set
        num_val_samples = int(len(X) * val_size)

        _logger.info(f"Splitting into train/validation with ratio: {1-val_size}/{val_size}")
        # Split the data into train and validation sets
        X_train = X[:-num_val_samples]
        y_train = y[:-num_val_samples]
        X_val = X[-num_val_samples:]
        y_val = y[-num_val_samples:]
        _logger.info(f"Train set size: {X_train.shape}")
        _logger.info(f"Validation set size: {X_val.shape}")
    
        return X_train, y_train, X_val, y_val

    def fit_model(self, 
                  X_train, 
                  y_train, 
                  X_val, 
                  y_val) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Build a NN model and fit the model

        Parameters
        ----------
        training_df: pyspark.sql.dataframe.DataFrame
            training set
        test_df: pyspark.sql.dataframe.DataFrame  
            test(validation) set
            
        Returns
        -------
        Tuple[tf.keras.Model, tf.keras.callbacks.History]:
            A tuple containing the trained Keras model and a dictionary of training history, 
            containing loss and metrics values for each epoch.
        """
        _logger.info('Creating Keras Model')

        _logger.info(f"Training parameters: {self.cfg.train_instructions}")
        model_trainer = ModelTrainer(self.cfg.train_instructions)

        model, model_history = model_trainer.train_model(X_train, y_train, X_val, y_val)

        return model, model_history
 
    def run(self):
        """
        run model training as follow:
            1. Set MLflow experiment (creating a new experiment if it does not already exist)
            2. Start MLflow run
            3. Create Databricks Feature Store training set
            4. Create train-test splits 
            5. Initiate model pipeline using ModelTrainingPipeline, and fit on train data and eventually get score results
            6. Log trained model using the Databricks Feature Store API. Model will be logged to MLflow with associated
               feature table metadata.
            7. Register the model to MLflow model registry if model_name is provided in mlflow_params
        """
        _logger.info("==========Running model training==========")
        mlflow_tracking_cfg = self.cfg.mlflow_tracking_cfg

        _logger.info("==========Setting MLflow experiment==========")
        #self._set_experiment(mlflow_tracking_cfg)
        
        # Enable automatic logging of metrics, parameters
        mlflow.tensorflow.autolog(log_models=False)
        
        _logger.info("==========Starting MLflow run==========")
        with mlflow.start_run(run_name=mlflow_tracking_cfg.run_name) as mlflow_run:

            if self.cfg.pipeline_config is not None:
                # Log config file
                mlflow.log_dict(self.cfg.pipeline_config, 'pipeline_config.yml')
                
            # log unique local dbfs storage path 
            mlflow.log_param("dbfs_local_storage_path", self.dbfs_local_storage_path)
            mlflow.log_param("max_features", 
                             self.cfg.feature_transformation_cfg.tokenization_instructions.get("max_features"))
            mlflow.log_param("max_text_len", 
                             self.cfg.feature_transformation_cfg.padding_instructions.get("max_text_len"))
      
            # get mlflow artifact uri
            artifact_uri = mlflow_run.info.artifact_uri

            # Create Feature Store Training Set
            _logger.info("==========Loading Training Features & Labels==========")
            train_padded_corpus, y_train = self.run_feature_preparation_for_training()
            
            # Load and preprocess data into train/test splits
            _logger.info("==========Creating Train/Validation Splits==========")
            X_train, y_train, X_val, y_val = self.train_validation_split(train_padded_corpus, 
                                                                         y_train, 
                                                                         self.cfg.train_val_split_config.get("validation_size")) 
            # Log data related paramters
            mlflow.log_param("train_dataset_shape", X_train.shape)
            mlflow.log_param("train_label_shape", y_train.shape)
            mlflow.log_param("validataion_dataset_shape", X_val.shape)
            mlflow.log_param("validation_label_shape", y_val.shape)
            mlflow.log_param("handel_imbalanced_labels_method", self.handel_imbalanced_labels_method)
            
            # assign number of claasses as output and log dropout
            self.cfg.train_instructions["num_classes"] = y_train.shape[1]
            mlflow.log_param("dropout", self.cfg.train_instructions.get("dropout"))
            
            # Fit pipeline with Classifier and log model
            _logger.info("==========Fitting Classifier Model==========")
            model, model_history = self.fit_model(X_train, y_train, X_val, y_val)
            mlflow.keras.log_model(model, "model")
            
            # Get evaluation resutls of the test
            _logger.info("==========Model Evaluation On Validation Data==========")
            _logger.info("Evaluating and logging accuracy, precision, recall and f1-score metrics")           
            # Log other metrics using validation dataset
            y_pred = model.predict(X_val)
            precision, recall, f1_score, _ = precision_recall_fscore_support(y_val.argmax(axis=-1), 
                                                                             y_pred.argmax(axis=-1), 
                                                                             average='weighted')
            mlflow.log_metrics({'val_precision': round(precision, 3), 
                                'val_recall': round(recall, 3), 
                                'val_f1_score': round(f1_score, 3)
                               })

            _logger.info('==========Logging Mlflow Artifcats==========')
            tokenizer_artifact_path = self.dbfs_local_storage_path + "tokenizer.pkl"
            mlflow.log_artifact(tokenizer_artifact_path)
            _logger.info('Tokenizer logged as an artifact')
            
            label_encoder_artifact_path = self.dbfs_local_storage_path + "label_encoder.pkl"
            mlflow.log_artifact(label_encoder_artifact_path)
            _logger.info('Label Encoder logged as an artifact')
            
            if self.infrequent_labels_pdf is not None:
                infrequent_labels_artifact_path = self.dbfs_local_storage_path + "infrequent_labels.csv"
                mlflow.log_artifact(infrequent_labels_artifact_path)
                _logger.info('Infrequent labels logged as an artifact')
            
            log_acc_and_loss_plot(model_history, self.dbfs_local_storage_path)
            mlflow.log_artifact(self.dbfs_local_storage_path + "train_validation_loss_and_accuracy.svg")
            _logger.info('Train/Validation Acurracy/Loss plot logged as an artifact')
                
            # Create a PyFunc model that uses the trained Keras model and label encoder
            pyfunc_model = KerasModel(model, self.tokenizer, self.label_encoder)
            mlflow.pyfunc.log_model("custom_model", python_model= pyfunc_model)
            
            model_uri = artifact_uri + "/custom_model"
            # Register model to MLflow Model Registry if provided
            if mlflow_tracking_cfg.model_name is not None:
                _logger.info("==========MLflow Model Registry==========")
                if mlflow_tracking_cfg.registry_uri:
                    _logger.info(f"The registry_uri uri is: {mlflow_tracking_cfg.registry_uri}")
                    mlflow.set_registry_uri(mlflow_tracking_cfg.registry_uri)
                _logger.info(f"Registering model: {mlflow_tracking_cfg.model_name}")
                mlflow.register_model(model_uri, name=mlflow_tracking_cfg.model_name)

            _logger.info("==========Training, Evaluation & Model Registeration Completed!==========")
            