"""
listing1_lstm_model.py
======================
Listing 1 – LSTM-Based Intrusion Classifier
--------------------------------------------
Full Keras implementation of the two-layer LSTM model used in:

  "A Lightweight Hybrid Intrusion Detection System (IDS)
   for Edge Network Security"

Companion to Algorithm 1 in the paper body.

Dependencies
------------
    pip install tensorflow numpy

Usage
-----
    from listing1_lstm_model import create_lstm_model
    model = create_lstm_model(input_shape=(1, 78), num_classes=12)
    model.summary()
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout
)


def create_lstm_model(input_shape, num_classes):
    """
    Build and compile a two-layer stacked LSTM classifier.

    Parameters
    ----------
    input_shape : tuple
        Shape of one input sample, e.g. (1, 78) for a single
        time-step with 78 network-flow features.
    num_classes : int
        Number of output classes (attack categories + benign).
        In the paper experiments: 12 (11 attack types + BENIGN).

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for model.fit().

    Architecture
    ------------
    Input (1, 78)
        └─ LSTM(128, return_sequences=True)
        └─ Dropout(0.3)
        └─ LSTM(64,  return_sequences=False)
        └─ Dropout(0.3)
        └─ Dense(64,  ReLU)
        └─ Dropout(0.3)
        └─ Dense(32,  ReLU)
        └─ Dropout(0.2)
        └─ Dense(num_classes, Softmax)   ← output
    """
    model = Sequential([
        Input(shape=input_shape),

        # ── Recurrent layers ──────────────────────────────
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),

        # ── Fully-connected head ──────────────────────────
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),

        # ── Softmax output ────────────────────────────────
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# ── Example usage ─────────────────────────────────────────────
if __name__ == '__main__':
    NUM_CLASSES = 12       # 11 attack categories + BENIGN
    INPUT_SHAPE = (1, 78)  # 1 time-step, 78 flow features

    model = create_lstm_model(INPUT_SHAPE, NUM_CLASSES)
    model.summary()
