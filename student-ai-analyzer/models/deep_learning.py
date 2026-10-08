"""
models/deep_learning.py
------------------------
Optional Deep Learning demonstration using TensorFlow / Keras.

This module is intentionally kept very simple:
  - Input layer  (8 features)
  - 1 hidden Dense layer (16 neurons, ReLU)
  - Output layer (4 neurons, Softmax → 4 career classes)

If TensorFlow is not installed, the module raises an ImportError
which the Streamlit page catches and shows a friendly message.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from utils.preprocessing import load_student_data, FEATURE_COLS, TARGET_COL

# TensorFlow / Keras – optional
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


def build_and_train(epochs: int = 30):
    """
    Build and train a small neural network on the student dataset.

    Returns
    -------
    history        : Keras History object (loss & accuracy per epoch)
    test_accuracy  : float
    class_names    : list of career label strings
    TF_AVAILABLE   : bool (so caller can check)

    Raises ImportError if TensorFlow is not installed.
    """
    if not TF_AVAILABLE:
        raise ImportError(
            "TensorFlow is not installed. Install it with: pip install tensorflow"
        )

    df = load_student_data()

    X = df[FEATURE_COLS].to_numpy(dtype="float32")
    y_raw = df[TARGET_COL].to_numpy()

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = list(le.classes_)
    num_classes = len(class_names)

    # Scale features to roughly zero-mean, unit-variance
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build the model
    model = keras.Sequential([
        # Input layer is implicit via input_shape
        keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],),
                           name="hidden_layer"),
        keras.layers.Dense(num_classes, activation="softmax", name="output_layer"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Train – verbose=0 keeps the Streamlit output clean
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        validation_data=(X_test, y_test),
        verbose=0,
    )

    # Final evaluation on the test set
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

    return history, round(float(test_accuracy) * 100, 2), class_names
