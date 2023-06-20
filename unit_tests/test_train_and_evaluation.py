import pytest
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from mock import MagicMock, patch, Mock

from keras.layers import Dense, Dropout
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense 

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

from modules.utils.train_and_evaluation import ModelTrainer, log_acc_and_loss_plot
  
########################################################################################
## Fixtures
########################################################################################  
@pytest.fixture
def vocab_size():
    return 100
    
@pytest.fixture
def max_text_len():
    return 5

@pytest.fixture
def num_classes():
    return 4

@pytest.fixture
def train_instructions(vocab_size, max_text_len):
    train_instructions = {
            "dropout": "0.3",
            "input_dim": vocab_size,
            "input_length": max_text_len,
            "compile_conf": {
                "metrics": ["accuracy"]
            },
            "callbacks": {
                "early_stop": {
                    "monitor": "val_accuracy",
                    "mode": "max",
                    "patience": 3
                }
            },
            "fit_conf": {
                "epochs": 5,
                "batch_size": 512,
                "shuffle": True,
                "verbose": 1
            }
        }

    return train_instructions

########################################################################################
## Function Tests
########################################################################################   
def test_build_model(train_instructions, num_classes):
    train_instructions['num_classes'] = 4
    model_trainer = ModelTrainer(train_instructions)
    model = model_trainer._build_model(dropout=0.3)
    assert isinstance(model, tf.keras.Model)
    assert len(model.layers) == 6
    isinstance(train_instructions['dropout'], float)
  
    assert isinstance(model.layers[0], Embedding)
    assert isinstance(model.layers[1], Conv1D)
    assert isinstance(model.layers[2], GlobalMaxPooling1D)
    assert isinstance(model.layers[3], Dense)
    assert isinstance(model.layers[4], Dropout)
    assert isinstance(model.layers[5], Dense)

def test_fit_model(train_instructions, vocab_size, max_text_len, num_classes):
  
    # Generate mock training data
    num_samples_train = 20
    num_samples_val = 4
    
    # paramter of the input shape to the model
    input_length = max_text_len
    input_dim = vocab_size
    
    # Generate mock train data
    X_train = np.random.randint(1, input_dim, size=(num_samples_train, input_length))
    y_train = np.random.randint(0, num_classes, size=num_samples_train)

    X_train = pad_sequences(X_train, maxlen=input_length)
    y_train = to_categorical(y_train, num_classes=num_classes)

    # Generate mock validation data
    X_val = np.random.randint(1, input_dim, size=(num_samples_val, input_length))
    y_val = np.random.randint(0, num_classes, size=num_samples_val)

    X_val = pad_sequences(X_val, maxlen=input_length)
    y_val = to_categorical(y_val, num_classes=num_classes)

    train_instructions['num_classes'] = num_classes


    model_trainer = ModelTrainer(train_instructions)
    model, history = model_trainer.train_model(X_train, y_train, X_val, y_val)  
    
    assert isinstance(model, tf.keras.Model)
    assert isinstance(history, tf.keras.callbacks.History)
    assert 1==1
    