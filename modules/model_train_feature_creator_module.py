from typing import Union, Dict, Any

import numpy as np
import pandas as pd

import pyspark.sql.dataframe
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from pyspark.sql import functions as f

from modules.utils import featurize
from modules.utils.logger_utils import get_logger
from modules.utils.persist_and_load_utils import load_table, persist_to_delta_table
from modules.utils.common import (IntermediateTransformationConfig, IntermediateTableConfig, 
                        FeatureTableCreatorConfig)
from modules.utils.spark_utils import (add_prefix_to_columns, filter_df_from_instructions, 
                                   union_dfs, leftsemi_filtering)


class TrainFeatureCreator:
    """
    Class to execute a pipeline to create intermediate table, and persist it to
    the provided intermediate storage path
    """
    def __init__(self, cfg: FeatureTableCreatorConfig):
        global _logger
        global spark
        _logger = get_logger()
        spark = SparkSession.builder.getOrCreate()
        self.cfg = cfg
    
    def _run_data_ingest(self, table_name: str) -> pyspark.sql.DataFrame:
        """
        Run data ingest step
        
        Parameters
        -------
        table_name: str
            name of the given table to be ingested
            
        Returns
        -------
        pyspark.sql.DataFrame
            Ingested Spark DataFrame
        """
        
        # path to the loading table
        table_path = self.cfg.input_tables.get(table_name).get("path")

        ingested_df = load_table(table_path, "csv", spark)
            
        if self.cfg.input_tables.get(table_name).get("prefix"):
            ingested_df = add_prefix_to_columns(ingested_df, self.cfg.input_tables.get(table_name).get("prefix"))
        
        _logger.info(f'Table {table_name} with dimension {(ingested_df.count(), len(ingested_df.columns))} ingested successfully!')    
        
        return ingested_df 
    
    def _load_and_union_community_tables(self):

        # get the name of community tables into a list
        all_table_names = self.cfg.input_tables.keys()
        community_table_names = [table_name for table_name in all_table_names if "production_2022_community_" in table_name]
        
        dfs_list = []
        for table_name in community_table_names:    
            dfs_list.append(self._run_data_ingest(table_name))
                
        unioned_df = union_dfs(*dfs_list)    

        _logger.info(f'Union of the following tables has been successful!: {community_table_names}')
        _logger.info(f'Dimnesion of the unioned table: {(unioned_df.count(), len(unioned_df.columns))}')
        
        return unioned_df
    
    def _load_unspsc_referenc_table(self):
        if "unspsc_reference_table" in self.cfg.input_tables.keys():
            
            ref_table_name = self.cfg.input_tables.get("unspsc_reference_table")
            if ref_table_name:
                return self._run_data_ingest("unspsc_reference_table")
            else: 
                return None
    
    @staticmethod
    def filter_and_split_legit_unspsc(df: pyspark.sql.dataframe.DataFrame, 
                                      unspsc_reference_df: Union[pyspark.sql.DataFrame, None],
                                      unspsc_filter_instructions: Dict[str, Any]) -> pyspark.sql.dataframe.DataFrame:
        """
        Take a pyspark.DataFrame 
            1. Filter UNSPSC that contains 8 digits format and all digits are not zeros (e.g. "0000000")
            2. Split UNSPCS 8 digits columns to 4 sub categories and add those columns to dataframe
                 - SEGMENT
                 - FAMILY
                 - CLASS
                 - COMMODITY
            3. Filter the rows in which one of the SEGMENT, FAMILY or CLASS are zeros 
               and do not belong to any criteria (e.g. SEGMENT = "00")

        Parameters
        ----------
        df: pyspark.sql.dataframe.DataFrame
            given dataframe
        unspsc_reference_df: Union[pyspark.sql.DataFrame, None]
            unspsc reference table
        handle_na_instructions: Dict[str, Any]
            instructions to filter legit UNSPSC 

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            filtered dataframe
        """
        unspsc_col = unspsc_filter_instructions.get("unspsc_col")
        length_filter_instruction = unspsc_filter_instructions.get("length_filter")
        non_zero_filter_instruction = unspsc_filter_instructions.get("non_zero_filter")
        
        # get 8 digits and filter out all-zero unspsc
        df_filtered_by_length = filter_df_from_instructions(df, length_filter_instruction)
        
        # drop NAN unspsc
        df_filtered_by_length = df_filtered_by_length.dropna(subset=unspsc_col, how="any")
        
        # split unspsc columns to sub-categories 
        df_filtered_with_sub_categories = (df_filtered_by_length.withColumn("SEGMENT", f.substring(f.col(unspsc_col), 1, 2))
                                                                .withColumn("FAMILY", f.substring(f.col(unspsc_col), 3, 2))
                                                                .withColumn("CLASS", f.substring(f.col(unspsc_col), 5, 2))
                                                                .withColumn("COMMODITY", f.substring(f.col(unspsc_col), 7, 2))
                                          )
        # filter non-zeros segments, family and class
        df_with_legit_unspsc = filter_df_from_instructions(df_filtered_with_sub_categories, non_zero_filter_instruction)
        
        # filter unspsc using refrence table, if provided
        if unspsc_reference_df:
            df_with_legit_unspsc = leftsemi_filtering(df_with_legit_unspsc, unspsc_reference_df, unspsc_col, unspsc_col)
            
        return df_with_legit_unspsc
    
    def run_raw_data_prep(self, input_df: pyspark.sql.DataFrame, 
                      unspsc_reference_df: Union[pyspark.sql.DataFrame, None]) -> pyspark.sql.DataFrame:
        """
        Run data preparation step on the raw data, using Featurizer to run featurization logic to create 
        intermediate table from the input DataFrame.

        Parameters
        ----------
        input_df: pyspark.sql.DataFrame
            Input Spark DataFrame
        unspsc_reference_df: Union[pyspark.sql.DataFrame, None]
            unspsc reference table
            
        Returns
        -------
        pyspark.sql.DataFrame
            Processed Spark DataFrame containing preprocessed data intermediate table
        """
        # Get legit UNSPSC and after filteration delete key from config
        _logger.info("Filtering legit UNSPSCs")
        df_with_legit_unspsc = self.filter_and_split_legit_unspsc(input_df, unspsc_reference_df, self.cfg.intermediate_transformation_cfg.unspsc_processsing)
        del self.cfg.intermediate_transformation_cfg.unspsc_processsing
        
        featurizer = featurize.Featurizer(self.cfg.intermediate_transformation_cfg)
        processed_df = featurizer.run(df_with_legit_unspsc)
        
        return processed_df

    def persist_train_to_intermediate_table(self, df: pyspark.sql.DataFrame) -> None:
        """
        Method to create and persist intermediate table in given storage path. When run, this method will create from scratch the
        table. As such, we first create (if it doesn't exist) the database specified, and drop the table if it
        already exists.

        Parameters
        ----------
        df: pyspark.sql.DataFrame
            Spark DataFrame from which to create the intermediate table.
        """
        int_table_cfg = self.cfg.intermediate_table_cfg
 
        _logger.info(f"Persisting intermediate table to the given storage at: {int_table_cfg.storage_path}")
        persist_to_delta_table(df, int_table_cfg.storage_path)
        #persist_to_delta_table(df, int_table_cfg.database_name, int_table_cfg.table_name, int_table_cfg.storage_path, spark)
        #_logger.info(f"Created intermediate database: {int_table_cfg.database_name}.{int_table_cfg.table_name}")
        _logger.info(f"Delta table has been created successfully at {int_table_cfg.storage_path}")
    
    def run(self) -> None:
        """
        Run data preprocessing pipeline
        """
        _logger.info('Initiated Intermediate transformation ....')
        _logger.info('==========Data Ingestion & Join==========')
        input_df = self._load_and_union_community_tables()
        unspsc_reference_df = self._load_unspsc_referenc_table()

        _logger.info('==========Data Prep==========')
        processed_df = self.run_raw_data_prep(input_df, unspsc_reference_df)

        _logger.info('==========Persist Intermediate Table==========')
        self.persist_train_to_intermediate_table(processed_df)
        
 


      
            
            
            
            
            
            
            
            
            
            
            
            
        