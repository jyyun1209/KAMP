import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras


def _set_seed(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _build_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(units=64, return_sequences=False),
        #keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(x_train, y_train, x_valid=None, y_valid=None,
          epochs=300, batch_size=256, seed=0,
          patience=50, save_path="model_best_rnn.keras"):
    _set_seed(seed)
    model = _build_model(x_train.shape[1:])

    val_data = (x_valid, y_valid) if x_valid is not None else None

    callbacks = []
    if val_data is not None:
        callbacks.append(keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ))
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=max(1, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ))
        if save_path is not None:
            callbacks.append(keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
            ))

    history = model.fit(
        x_train, y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history.history


def predict(model, x):
    proba = model.predict(x, verbose=0)
    return np.argmax(proba, axis=1)


def predict_proba(model, x):
    return model.predict(x, verbose=0)[:, 1]
