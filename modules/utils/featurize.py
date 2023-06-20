from typing import Any, Dict, Union
from dataclasses import dataclass

import pyspark
import pyspark.pandas as ps
from pyspark.sql import functions as f

from modules.utils.logger_utils import get_logger
from modules.utils.common import IntermediateTransformationConfig
from modules.utils.spark_utils import filter_df_from_instructions, replace_columns_values, leftsemi_filtering

_logger = get_logger()


class Featurizer:
    """
    Class containing featurization logic to apply to input Spark DataFrame
    """

    def __init__(self, cfg: IntermediateTransformationConfig):
        self.cfg = cfg

    @staticmethod
    def select_columns(df: pyspark.sql.dataframe.DataFrame,
                       select_columns_instruction: Dict[str, Any]) -> pyspark.sql.dataframe.DataFrame:
        """
        Take a pyspark.DataFrame and select a subset of columns and returns the dataframe

        Parameters
        ----------
        df: pyspark.sql.dataframe.DataFrame
            given dataframe
        select_columns_instruction: Dict[str, Any] 
             instructions for column selection including list of columns
        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            filtered dataframe
        """

        return filter_df_from_instructions(df, select_columns_instruction)

    @staticmethod
    def unify_and_filter_na_values(df: pyspark.sql.dataframe.DataFrame,
                                   handle_str_na_instructions: Dict[str, Any]) -> pyspark.sql.dataframe.DataFrame:
        """
        Take a pyspark.DataFrame 
            1. unify different format of NAs into single format 
            2. Filter the rows of the columns that contains 
               that specific format

        Parameters
        ----------
        df : pyspark.sql.dataframe.DataFrame
            given dataframe
        handle_str_na_instructions: Dict[str, Any]
            instructions to unify and filter different 
            format of string NA values 

        Returns
        -------
        pyspark.sql.dataframe.DataFrame
            filtered dataframe
        """
        # get configurations from dictionary
        columns = handle_str_na_instructions.get("columns")
        na_vals_list = handle_str_na_instructions.get("na_vals_list")
        replacement_val = handle_str_na_instructions.get("replacement_val")
        na_filter_instructions = handle_str_na_instructions.get("na_filter_instructions")

        # unify NA values to N/A and 
        df_na_replaced = replace_columns_values(df, columns, na_vals_list, replacement_val)

        # filter out the rows that contains N/A in the given columns
        filtered_df = filter_df_from_instructions(df_na_replaced, na_filter_instructions)

        return filtered_df

    @staticmethod
    def drop_duplicates(df: pyspark.sql.dataframe.DataFrame,
                        drop_duplicates_instructions: Dict[Any, Any]) -> pyspark.sql.dataframe.DataFrame:
        """
        Drops duplicate rows in two parts:
            1. get unique PO_ITEM_ID as index for feature store
            2. get unique rows based on the selected subset of columns
        return a dataframe with unique rows

        Parameters
        ----------
        df:  pyspark.sql.dataframe.DataFrame
            The DataFrame from which to drop rows.

        drop_duplicates_instructions: Dict[Any, Any]
            configuration of how to handle missing values

        Returns
        ----------
        pyspark.sql.dataframe.DataFrame
            The modified DataFrame with rows containing null or missing values removed.
        """
        df_with_unique_index = df.dropDuplicates(drop_duplicates_instructions.get("index_col"))
        df_with_unique_rows = df_with_unique_index.dropDuplicates(drop_duplicates_instructions.get("cat_cols"))

        _logger.info(f"number of dropped rows: {df.count() - df_with_unique_rows.count()}")

        return df_with_unique_rows

    @staticmethod
    def drop_missing_values(df: pyspark.sql.dataframe.DataFrame,
                            dropna_config: Dict[Any, Any]) -> pyspark.sql.dataframe.DataFrame:
        """
        Drops rows containing null or missing values from the given DataFrame

        Parameters
        ----------
        df:  pyspark.sql.dataframe.DataFrame
            The DataFrame from which to drop rows.
        dropna_config: Dict[Any, Any]
            configuration of how to drop missing values

        Returns
        ----------
        pyspark.sql.dataframe.DataFrame
            The modified DataFrame with rows containing null or missing values removed.
        """
        if dropna_config:
            return df.dropna(**dropna_config)
        else:
            return df.dropna()

    @staticmethod
    def fill_missing_values(df: pyspark.sql.dataframe.DataFrame,
                            fillna_config: Dict[Any, Any]) -> pyspark.sql.dataframe.DataFrame:
        """
        Fills rows containing null or missing values

        Parameters
        ----------
        df:  pyspark.sql.dataframe.DataFrame
            The DataFrame in which NAN rows gets filled.
        fillna_config: Dict[Any, Any]
            configuration of how to fill missing values

        Returns
        ----------
        pyspark.sql.dataframe.DataFrame
            The modified DataFrame with NA rows filled with given alternative values
        """
        if fillna_config:
            return df.fillna(**fillna_config)
        else:
            return df.fillna()

    def run(self, df: pyspark.sql.DataFrame) -> pyspark.sql.DataFrame:
        """
        Run all data preprocessing steps. Consists of the following:

            1. Convert PySpark DataFrame to pandas_on_spark DataFrame 
            2. Drop any missing values if specified in the config
            3. Return resulting preprocessed dataset as a PySpark DataFrame

        Parameters
        ----------
        df : pyspark.sql.DataFrame
            Input PySpark DataFrame to preprocess

        Returns
        -------
        pyspark.sql.DataFrame
            Preprocessed dataset of features
        """
        _logger.info('Running Data Preprocessing steps...')
        # Select columns
        _logger.info("Selecting columns and filter NAs")
        df_with_selected_columns = self.select_columns(df, self.cfg.select_cols_instructions)
        _logger.info(f"After select columns: {df_with_selected_columns.count()}")

        # Unify the format of NAs and filter rows
        _logger.info("Unifying NA formats to N/A and filtering N/A rows")
        df_na_replaced_and_filtered = self.unify_and_filter_na_values(df_with_selected_columns,
                                                                      self.cfg.handle_str_na_vals)
        _logger.info(f"After unifying na: {df_na_replaced_and_filtered.count()}")

        # Drop duplicates from index column and categorical rows
        _logger.info("Dropping duplicates")
        df_preprocessed = self.drop_duplicates(df_na_replaced_and_filtered,
                                               self.cfg.drop_duplicates_instructions)
        _logger.info(f"After drop duplicates: {df_preprocessed.count()}")

        # Drop missing values
        if self.cfg.handle_missing_vals.get("drop_na_instructions").get("drop_rows"):
            _logger.info("Dropping missing values")
            df_preprocessed = self.drop_missing_values(df_preprocessed,
                                                       self.cfg.handle_missing_vals.get("drop_na_instructions").get(
                                                           "dropna_config"))
            _logger.info(f"After drop missing values: {df_preprocessed.count()}")

        if self.cfg.handle_missing_vals.get("fill_na_instructions").get("fill_missing_val"):
            _logger.info("Filling missing values")
            df_preprocessed = self.fill_missing_values(df_preprocessed,
                                                       self.cfg.handle_missing_vals.get("fill_na_instructions").get(
                                                           "fillna_config"))
            _logger.info(f"After fill missing values: {df_preprocessed.count()}")
        return df_preprocessed