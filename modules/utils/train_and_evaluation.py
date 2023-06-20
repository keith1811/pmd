from typing import Dict, Tuple

import matplotlib.pyplot as plt

import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Dropout
from keras.preprocessing.text import one_hot
from keras.losses import categorical_crossentropy
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping


class ModelTrainer:
    def __init__(self, train_instructions: dict):
        self.train_instructions = train_instructions
        
        if self.train_instructions.get("dropout"):
            self.model = self._build_model(float(self.train_instructions.get("dropout")))
        else:
            self.model = self._build_model()
        
    def _build_model(self, dropout=0.0) -> tf.keras.Model:
        """
        Builds a convolutional neural network (CNN) model for text classification using Keras.

        Parameters
        ----------
            dropout: float
                the dropout rate to apply to the fully connected layer (default 0.0).

        Returns
        -------
            tf.keras.Model
                A Keras Model object representing the CNN model
        """
        model = Sequential()
        
        model.add(Embedding(int(self.train_instructions.get("input_dim")), 
                            100, 
                            input_length=int(self.train_instructions.get("input_length"))))
        
        model.add(Conv1D(64, 5, activation='relu'))

        model.add(GlobalMaxPooling1D())

        model.add(Dense(units=256, activation="relu"))

        model.add(Dropout(dropout))

        model.add(Dense(units=int(self.train_instructions.get("num_classes")),
                        activation="softmax"))

        return model

    def train_model(self, 
                    X_train: tf.Tensor, 
                    y_train: tf.Tensor, 
                    X_val: tf.Tensor, 
                    y_val: tf.Tensor) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
        """
        Trains the Keras model using the provided training and validation data.

        Parameters
        ----------
        X_train: tf.Tensor
            Input data for training.
        y_train: tf.Tensor
            Target data for training.
        X_val: tf.Tensor
            Input data for validation.
        y_val: tf.Tensor
            Target data for validation.

        Returns
        -------
            Tuple[tf.keras.Model, tf.keras.callbacks.History]
                A tuple containing the trained Keras model and a dictionary of training history containing loss and metrics values for each epoch.
        """
        if self.train_instructions.get("compile_conf"):
            
            if self.train_instructions.get("compile_conf").get("optimzer_conf"):
                optimizer = Adam(self.train_instructions.get("compile_conf").get("optimzer_conf"))
            else:
                optimizer = Adam()
                
            metrics = self.train_instructions.get("compile_conf").get("metrics")
            self.model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=metrics)
        else:
            self.model.compile()

        # Set up callbacks
        callbacks = []
        if self.train_instructions.get("callbacks").get("tensorboard"):
            callbacks.append(TensorBoard(**self.train_instructions.get("callbacks").get("tensorboard")))
        if self.train_instructions.get("callbacks").get("checkpoint"):
            callbacks.append(ModelCheckpoint(**self.train_instructions.get("callbacks").get("checkpoint")))
        if self.train_instructions.get("callbacks").get("early_stop"):
            callbacks.append(EarlyStopping(**self.train_instructions.get("callbacks").get("early_stop")))

        # Train the model with callbacks
        model_history = self.model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=callbacks, **self.train_instructions.get("fit_conf"))

        return self.model, model_history

    
def log_acc_and_loss_plot(model_history, path_to_dir: str) -> None:
    """
    Trains the model using the provided training data and validation data, and returns the trained model and the model history.

    Parameters
    ----------
    X_train: tf.Tensor
        The training data to fit the model to.
    y_train: tf.Tensor
        The target values for the training data.
    X_val: tf.Tensor
        The validation data to validate the model on.
    y_val: tf.Tensor
        The target values for the validation data.

    Returns
    -------
    Tuple[tf.keras.Model, Dict]
        The trained Keras model and a dictionary containing the training history.
    """
    fig, axe1 = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    axe1[0].plot(
        model_history.history["accuracy"],
        label="train accuracy",
        color="blue",
        marker='.'
    )
    axe1[0].plot(
        model_history.history["val_accuracy"],
        label="val accuracy",
        color="red",
        marker='.'
    )

    axe1[1].plot(
        model_history.history["loss"],
        label="train loss",
        color="blue",
        marker='*'
    )
    axe1[1].plot(
        model_history.history["val_loss"],
        label="val loss",
        color="red",
        marker='*'
    )

    axe1[0].title.set_text("CNN Training vs. Validation Accuracy")
    axe1[1].title.set_text("CNN Training vs. Validation Loss")

    axe1[0].set_xlabel("Epoch")
    axe1[1].set_xlabel("Epoch")

    axe1[0].set_ylabel("Rate")
    axe1[1].set_ylabel("Rate")

    axe1[0].legend(['train', 'val'], loc='upper left')
    axe1[1].legend(['train', 'val'], loc='upper left')
    fig.savefig(path_to_dir + "train_validation_loss_and_accuracy.svg")
   
    plt.show(block=False)
    plt.pause(3)
    plt.close()

