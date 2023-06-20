from modules.utils.config_utils import get_feature_concat_cols, get_feature_concat_name
import numpy as np
import pandas as pd
from pyspark.sql import functions as f
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer 

def add_concat_feature_column_to_df(df, env, pipeline_name='model_inference') -> pd.DataFrame:
    feature_cols_list = get_feature_concat_cols(env, pipeline_name=pipeline_name)
    concat_col_name = get_feature_concat_name(env, pipeline_name=pipeline_name)
    df = df.withColumn(concat_col_name, f.concat_ws(" ", *feature_cols_list))
    return df

def prepare_corpus_for_model(df: pd.DataFrame, 
                             ohe_column: str ,
                             vocab_size: int =20001, 
                             tokenizer: Tokenizer = None) -> list:
    """
    Preprocesses text data for use in the model.

    Parameters:
    -----------
    df: pandas DataFrame
        The dataset containing text data.
    ohe_column: str
        name of the column to be one-hot-encoded
    max_features: int, optional (default=20001)
        The maximum number of unique words to be included in the one-hot encoding.

    Returns:
    --------
    list:
        A list datasets, after one-hot encoding.
    """
    if not tokenizer:
        # Convert text data to numerical sequences
        tokenizer = Tokenizer(
                              num_words=vocab_size, 
                              filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', 
                              lower=True
                             )
        # Fit tokenizer on train data
        tokenizer.fit_on_texts(df[ohe_column].values)

    # Transform train data
    X_train_seq = tokenizer.texts_to_sequences(df[ohe_column].values)

    return X_train_seq, tokenizer

def pad_corpus(ohe_corpus: list, max_text_len: int =300) -> np.ndarray:
    """
    Pads the one-hot encoded training and testing datasets to a specified length.

    Parameters:
    -----------
    ohe_corpus: list
        The dataset after one-hot encoding.
    max_text_len: int, optional (default=300)
        The maximum length of each sentence after padding.

    Returns:
    --------
    np.ndarray
        A numpy array containing the padded datasets
    """

    padded_corpus = pad_sequences(ohe_corpus,
                                        maxlen=max_text_len,
                                        padding='post')

    return padded_corpus