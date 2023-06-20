import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer

from modules.utils.feature_transform_utils import *

########################################################################################
## Unittest Utils
########################################################################################  
def nested_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if isinstance(list1[i], list) and isinstance(list2[i], list):
            if not nested_lists_equal(list1[i], list2[i]):
                return False
        elif list1[i] != list2[i]:
            return False
    return True

########################################################################################
## Fixtures
######################################################################################## 
@pytest.fixture
def sample_dataframe_1():
    """Mock sample df 1.""" 
    return pd.DataFrame({
                        "CONCAT_FEATURE": ['this is a test', 'another test', 'and one more'],
                        "label": [0, 1, 0]})
   
@pytest.fixture
def tokenized_column():
    return "CONCAT_FEATURE"

@pytest.fixture
def vocab_size():
    return 100

@pytest.fixture
def tokenizer(sample_dataframe_1, vocab_size, tokenized_column):
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    tokenizer = Tokenizer(num_words=vocab_size, 
                          filters = filters, 
                          lower = True)
    
    tokenizer.fit_on_texts(sample_dataframe_1[tokenized_column].values)
    
    return tokenizer

@pytest.fixture
def max_text_len():
    return 5

@pytest.fixture
def exp_output_1():
    exp_output = [[2, 3, 4, 1], [5, 1], [6, 7, 8]]
    return exp_output 

@pytest.fixture
def exp_output_2():
    exp_output = [[2, 3, 4, 1, 0], [5, 1, 0, 0, 0], [6, 7, 8, 0, 0]]
    return np.array(exp_output)
########################################################################################
## Function Tests
########################################################################################
def test_prepare_corpus_for_model(sample_dataframe_1, 
                                  exp_output_1, 
                                  tokenized_column,  
                                  vocab_size):
    actual_output, _ = prepare_corpus_for_model(sample_dataframe_1, tokenized_column, vocab_size)

    # check the output
    assert nested_lists_equal(actual_output, exp_output_1)

def test_pad_corpus(exp_output_1, exp_output_2, max_text_len):
     
    # Actual output
    actual_output = pad_corpus(exp_output_1, max_text_len=max_text_len)

    assert isinstance(actual_output, np.ndarray)

    # Check that the shape of the output is correct
    assert actual_output.shape == exp_output_2.shape

    # Check that the output values are correct
    assert_array_equal(actual_output, exp_output_2)
















