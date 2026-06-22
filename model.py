import tensorflow as tf

from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import (
    TimeDistributed,
    GlobalAveragePooling2D,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.models import Sequential


def build_model():

    xception = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    xception.trainable = False

    model = Sequential([

        TimeDistributed(
            xception,
            input_shape=(20, 224, 224, 3)
        ),

        TimeDistributed(
            GlobalAveragePooling2D()
        ),

        LSTM(
            64,
            return_sequences=False
        ),

        Dense(
            256,
            activation="relu"
        ),

        Dropout(
            0.5
        ),

        Dense(
            64,
            activation="relu"
        ),

        Dropout(
            0.3
        ),

        Dense(
            1,
            activation="sigmoid"
        )

    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0001
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model