from typing import Tuple
import mlflow
from modules.utils.config_utils import get_feature_concat_name
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
import pyspark.sql.dataframe
from pyspark.sql import Window
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import current_timestamp
from pyspark.sql.types import StructType, StructField, DoubleType, StringType
from modules.utils import featurize
from modules.utils.common import ModelInferenceConfig
from modules.utils.logger_utils import get_logger
from modules.utils.persist_and_load_utils import load_table, persist_to_delta_table
from modules.utils.feature_transform_utils import add_concat_feature_column_to_df, prepare_corpus_for_model, pad_corpus
from modules.utils.persist_with_dbutils import check_model_stage_consistence,split_inference_dataframe,persist_to_bna_report


class ModelInference:
    """
    Class to execute a pipeline to create inference feature table, 
    and apply the trained model to it
    """
    def __init__(self, cfg: ModelInferenceConfig):
        self.cfg = cfg
        global _logger
        global spark
        global dbutils
        global sbnutils
        global batch_utils
        global sbn_constants
        if self.cfg.env == 'ml':
            spark = SparkSession.builder.getOrCreate()
            dbutils = DBUtils(spark)
            _logger = get_logger()
        else:
            import utils.sbnutils
            import utils.batch_utils
            import utils.constants
            sbnutils = utils.sbnutils
            batch_utils = utils.batch_utils
            sbn_constants = utils.constants
            spark = utils.sbnutils.get_spark()
            dbutils = utils.sbnutils.get_dbutils()
            _logger = lambda: None
            _logger.info = utils.sbnutils.log_info
            global persist_to_ml_supplier_po_item
            from modules.utils.persist_with_dbutils import persist_to_ml_supplier_po_item as persist
            persist_to_ml_supplier_po_item = persist
    
    def setup_output_dir(self) -> None:
        """
        Create an empty output directory
        """
        dbutils.fs.rm(self.cfg.inference_output_table_path, True)
        
        dbutils.fs.mkdirs(self.cfg.inference_output_table_path)
        _logger.info(f'Empty output directory created at: {self.cfg.inference_output_table_path}')

    def load_model(self) -> tf.keras.Model:
        """
        Load registered model from given model_uri, and registry uri, given the stage
        in model registry

        Returns
        -------
            tf.keras.Model
        """
        model_registry_stage = self.cfg.mlflow_tracking_cfg.model_registry_stage
        model_name = self.cfg.mlflow_tracking_cfg.model_name
        model_version = self.cfg.mlflow_tracking_cfg.model_version

        # create model uri
        model_uri = f'models:/{model_name}/{model_version}'

        # set registry uri, if it was given
        if self.cfg.mlflow_tracking_cfg.registry_uri:
            _logger.info(f"The registry uri is: {self.cfg.mlflow_tracking_cfg.registry_uri}")
            mlflow.set_registry_uri(self.cfg.mlflow_tracking_cfg.registry_uri)

        return mlflow.pyfunc.load_model(model_uri)
    
    def _run_data_ingest(self) -> pyspark.sql.DataFrame:
        """
        Run data ingest step from provided path given
        the input table type
             
        Returns
        -------
        pyspark.sql.DataFrame
            Ingested Spark DataFrame
        """
        # path to the loading table
        table_path = self.cfg.inference_input_table_path
        table_type = self.cfg.inference_input_table_type
        
        ingested_df = load_table(table_path, table_type, spark)
        
        if self.cfg.env != 'ml':
            timestamp_range = batch_utils.get_batch_timestamp_range()
            _logger.info(f"Batch starting time: {timestamp_range[0]}")
            _logger.info(f"Batch ending time: {timestamp_range[1]}")
            timestamp_range_condition = f"""
                    {sbn_constants.DELTA_UPDATED_FIELD} >= '{timestamp_range[0]}' AND
                    {sbn_constants.DELTA_UPDATED_FIELD} <= '{timestamp_range[1]}'
                """
            ingested_df = ingested_df.where(timestamp_range_condition)
        
        _logger.info(f'Infernece table with the dimension {(ingested_df.count(), len(ingested_df.columns))} ingested successfully!')    
        
        return ingested_df 
    
    def run_raw_data_prep(self, input_df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Run data preparation step on the raw data, using Featurizer to run featurization logic to create 
        intermediate table from the input DataFrame.

        Parameters
        ----------
        input_df: pyspark.sql.DataFrame
            Input Spark DataFrame
            
        Returns
        -------
        pyspark.sql.DataFrame
            Processed Spark DataFrame containing preprocessed data intermediate table
        """
        if not isinstance(input_df, pyspark.sql.DataFrame):
            raise TypeError(f'Expected input_df to be a PySpark dataframe, but received {type(input_df)}')
            
        featurizer = featurize.Featurizer(self.cfg.intermediate_transformation_cfg)
        processed_df = featurizer.run(input_df)
        
        return processed_df
    
    def run_basic_inference_preparation(self)-> Tuple[np.ndarray, 
                                                      tf.keras.Model, 
                                                      Tokenizer, 
                                                      LabelEncoder]:
        """
        A function that runs the following:
        1. Setup the given output directory for storing inference results
        2. Load model, and stored artifacts including label encoder and tokenizer
        3. Run basic preprocessing (intermediate processing)
        
        Returns
        -------
        Tuple[np.ndarray, tf.keras.Model, Tokenizer, LabelEncoder] 
             processed Dataframe, model, tokenizer, label encoder 
        """
        if self.cfg.env == 'ml':
            _logger.info('==========Prepare Output Directory==========')
            self.setup_output_dir()

        _logger.info('==========Loading Model & Artifacts From Model Registry==========')
        model = self.load_model()
        
        # Load artifacts, stored with model
        unwrapped_model = model.unwrap_python_model()
        label_encoder = unwrapped_model._load_label_encoder()
        tokenizer = unwrapped_model._load_tokenizer()

        _logger.info('==========Data Ingestion & Join==========')
        input_df = self._run_data_ingest()

        _logger.info('==========Data Prep==========')
        processed_df = self.run_raw_data_prep(input_df)

        return processed_df, model, tokenizer, label_encoder
    
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
        feature_concat_instructions = self.cfg.feature_transformation_cfg.feature_concat_instructions
        
        df_concatenated = df.select(f.concat_ws(" ", *feature_concat_instructions["concat_cols"]).alias(feature_concat_instructions["concat_col_name"]))
        _logger.info(f'''Following columns {feature_concat_instructions["concat_cols"]} concatenated to single '{feature_concat_instructions["concat_col_name"]}' column''')
        
        return df_concatenated
    
    def chunk_spark_table(self, df: pyspark.sql.DataFrame, chunk_size: int) -> pyspark.sql.DataFrame:
        """
        Break the given dataframe into partiotions, and yield the output (generator)

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Given Spark DataFrame

        Returns
        -------
        pyspark.sql.DataFrame
            Chunked spark DataFrame
        """
        # add row_index
        df = df.withColumn("row_index", f.monotonically_increasing_id())

        # Get the total number of rows in the DataFrame
        total_rows = df.count()

        # Calculate the number of chunks based on the desired chunk size
        num_chunks = int((total_rows + chunk_size - 1) / chunk_size)

        # Iterate over the chunks
        for i in range(num_chunks):
            # Calculate the starting and ending row indices for the chunk
            start_index = i * chunk_size
            end_index = min((i + 1) * chunk_size, total_rows)

            # Select the rows for the chunk and convert to a Pandas DataFrame
            chunk = df.filter((df.row_index >= start_index) & (df.row_index < end_index))
            
            # Drop row_index column from dataframe
            yield chunk.drop("row_index")
    
    def transform_spark_to_pandas(self, df: pyspark.sql.DataFrame) -> pd.DataFrame:
        """
        Run data sampling function to return a fraction of the given dataframe

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Given Spark DataFrame

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame
        """

        # Transform pyspark to pandas dataframe
        pdf = df.toPandas()
        
        return pdf 

    def run_inference_tokenization_and_padding(self, df: pd.DataFrame, tokenizer: Tokenizer) -> np.ndarray: 
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
        
        inference_seq, _ = prepare_corpus_for_model(df, tokenization_instructions.get("tokenized_col_name"), tokenizer = tokenizer)
        inference_padded_corpus = pad_corpus(inference_seq, padding_instructions.get("max_text_len"))
        
        _logger.info(f'Dimensions of corpus after one-hot-encoding and padding: {inference_padded_corpus.shape}')
        
        return inference_padded_corpus

    def run_prediction(self, inference_arr: np.ndarray, model: tf.keras.Model, label_encoder: LabelEncoder) -> np.ndarray:
        """
        Fetch and apply model from mlflow model registry and returns
        prediction class and their associated probabilities.

        Parameters
        ----------
        inference_arr: np.ndarray
            Input processed array to apply model and score
        model: tf.keras.Model
            Loded model from model registry
        label_encoder: LabelEncoder
            Loaded fitted label encoder
            
        Returns
        -------
        Tuple[list, list]:
            Two arrays containing the predicted class and their assocaite probabilites
        """
        _logger.info(f"Applying model to the batch")
        y_pred_prob = model.predict(inference_arr)
        
        # Get the predicted class for each example in the test data
        y_pred = np.argmax(y_pred_prob, axis=1).tolist()

        # Get the probability of the predicted class for each example in the test data
        prob_of_pred_label = [round(float(y_pred_prob[i][y_pred[i]]), 3) for i in range(len(y_pred))]
        
        # transform from categorical to label index
        y_pred_label = label_encoder.inverse_transform(y_pred)
        
        return y_pred_label, prob_of_pred_label

    @staticmethod
    def join_predict_to_original_df(df: pyspark.sql.DataFrame, 
                                    y_pred_label: list,
                                    prob_of_pred_label: list) -> pyspark.sql.DataFrame:
        """
        Add the prediction arrays to original dataframe

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Input processed array to apply model and score
        y_pred_label: list
            Predicted labels
        prob_of_pred_label: list
            Predicted probabilities
        Returns
        -------
        pyspark.sql.DataFrame:
            the output inference dataframe with prediction column
        """        
        # Join predicted labels and their probabilities to dataframe                    
        prediction_df = spark.createDataFrame([(l, p) for l, p in zip(y_pred_label, prob_of_pred_label)], 
                                 schema=StructType([
                                     StructField("predicted_label", StringType(), True),
                                     StructField("predicted_label_proba", DoubleType(), True)
                                 ]))
                                
        #add 'sequential' index and join both dataframe to get the final result
        df = df.withColumn("row_idx", f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))
        prediction_df = prediction_df.withColumn("row_idx", f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))
        
        final_df = (df.join(prediction_df, df.row_idx == prediction_df.row_idx)
                    .drop("row_idx")
                   )
        return final_df 
        
    def score_all(self):
        _logger.info('==========Scoring All Is Initiated==========')
        # get common prepared feature and the model
        processed_df, model, tokenizer, label_encoder = self.run_basic_inference_preparation()

        _logger.info('==========Concatenate Features To One Column==========')
        processed_df = add_concat_feature_column_to_df(processed_df, self.cfg.env)

        _logger.info('==========Split Need To Inference Dataframe==========')
        if processed_df.count() == 0: return
        processed_df = split_inference_dataframe(processed_df, self.cfg.mlflow_tracking_cfg)
        
        _logger.info('==========Select Concat Feature Column==========')
        concat_col_name = get_feature_concat_name(self.cfg.env)
        concatenated_features_df = processed_df.select(concat_col_name)

        _logger.info('==========Transform Spark To Pandas Datafram==========')
        pandas_df = self.transform_spark_to_pandas(concatenated_features_df)
 
        _logger.info('==========Transform Text Column Using One-Hot-Encoding & Padding==========')
        pandas_df_processed = self.run_inference_tokenization_and_padding(pandas_df, tokenizer)
        
        _logger.info('==========Predicting Labels==========')
        y_pred_label, prob_of_pred_label = self.run_prediction(pandas_df_processed, model, label_encoder) 
        
        # Add prediction column to original processed dataframe
        spark_df = self.join_predict_to_original_df(processed_df, y_pred_label, prob_of_pred_label)

        pandas_df = spark_df.toPandas()
        values = pandas_df.values.tolist()
        columns = pandas_df.columns.to_list()
        copy_df = spark.createDataFrame(values, columns)

        if self.cfg.save_output:
            _logger.info("==========Writing To Output Directory==========")
            if self.cfg.env == 'ml':
                persist_to_delta_table(spark_df, self.cfg.inference_output_table_path)
            else:
                persist_to_bna_report(spark_df, self.cfg.mlflow_tracking_cfg)
                persist_to_ml_supplier_po_item(copy_df, self.cfg.mlflow_tracking_cfg)
            _logger.info(f"Output directory path: {self.cfg.inference_output_table_path}")
        else:
            return spark_df
         
    def score_in_batch(self):
        _logger.info('==========Batch Scoring Is Initiated==========')
        # get common prepared feature and the model
        processed_df, model, tokenizer, label_encoder = self.run_basic_inference_preparation()

        _logger.info('==========Concatenate Features To One Column==========')
        processed_df = add_concat_feature_column_to_df(processed_df, self.cfg.env)

        _logger.info('==========Split Need To Inference Dataframe==========')
        if processed_df.count() == 0: return
        processed_df = split_inference_dataframe(processed_df, self.cfg.mlflow_tracking_cfg)

        batch_cnt= 1
        _logger.info('==========Transform Spark To Pandas Datafram In Batches==========')
        for chunk in self.chunk_spark_table(processed_df, self.cfg.batch_size):
            
            _logger.info('==========Select Concat Feature Column==========')
            concat_col_name = get_feature_concat_name(self.cfg.env)
            concatenated_features_chunked_df = chunk.select(concat_col_name)

            _logger.info('==========Transform Spark Chunks into Pandas==========')
            chunked_pdf = self.transform_spark_to_pandas(concatenated_features_chunked_df)

            _logger.info('==========Transform Chunk Text Column Using Tokenization & Padding==========')
            processed_chunked_pdf = self.run_inference_tokenization_and_padding(chunked_pdf, tokenizer)
            
            _logger.info('==========Predicting Labels==========')
            y_pred_label, prob_of_pred_label = self.run_prediction(processed_chunked_pdf, model, label_encoder)           
            
            # Add prediction column to original processed dataframe
            spark_chunk = self.join_predict_to_original_df(chunk, y_pred_label, prob_of_pred_label)
            pandas_df = spark_chunk.toPandas()
            values = pandas_df.values.tolist()
            columns = pandas_df.columns.to_list()
            copy_df = spark.createDataFrame(values, columns)

            _logger.info(f"==========Batch {batch_cnt} inference completed==========")
            if self.cfg.save_output:
                _logger.info(f"==========Writing Batch {batch_cnt} To Output Directory==========")
                if self.cfg.env == 'ml':
                    persist_to_delta_table(spark_chunk, self.cfg.inference_output_table_path, mode = "append")
                else:
                    persist_to_bna_report(spark_chunk, self.cfg.mlflow_tracking_cfg)
                    persist_to_ml_supplier_po_item(copy_df, self.cfg.mlflow_tracking_cfg)
                _logger.info(f"Output directory path: {self.cfg.inference_output_table_path}")
            else:
                _logger.info("Generator will not output any results nor store it!")
            
            batch_cnt+=1
            
    def run(self):
        """
        Function to run the batch or all-at-once inference
        """
        # 1. model info check, the following situations will raise Exception
        # - model stage not consistent with config file and model info table
        # - No model found in model info table
        _logger.info(f"cfg env: {self.cfg.env}")
        check_model_stage_consistence(self.cfg.mlflow_tracking_cfg)

        if self.cfg.batch_scoring in ["False", "false"]:
            infernece_df = self.score_all()
            _logger.info("==========Model inference completed==========")
            return infernece_df
            
        elif self.cfg.batch_scoring in ["True", "true"]:     
            self.score_in_batch()
            _logger.info("==========Batch Model inference completed==========")    
            
        else:
            _logger.info("==========Inference Could Not Start!==========")