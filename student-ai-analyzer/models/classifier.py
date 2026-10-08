"""
models/classifier.py
---------------------
Decision Tree Classifier for Student Career Prediction.

Workflow:
  Load data → split → train Decision Tree → evaluate → predict
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from utils.preprocessing import load_student_data, FEATURE_COLS, TARGET_COL


def train_classifier():
    """
    Train a Decision Tree Classifier on the student dataset.

    Returns
    -------
    model       : trained DecisionTreeClassifier
    X_test      : test feature matrix
    y_test      : true test labels
    y_pred      : predicted labels on test set
    accuracy    : float
    cm          : confusion matrix (ndarray)
    report      : classification report (str)
    feature_imp : dict  {feature_name: importance_value}
    classes     : list of class names
    """
    df = load_student_data()

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy()

    # Split – 80% train, 20% test, reproducible with random_state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision Tree – max_depth=5 keeps it readable and avoids overfitting
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42,
        criterion="gini",
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm       = confusion_matrix(y_test, y_pred, labels=model.classes_)
    report   = classification_report(y_test, y_pred, target_names=model.classes_)

    # Feature importances – pair each feature name with its importance score
    feature_imp = {
        name: round(float(imp), 4)
        for name, imp in zip(FEATURE_COLS, model.feature_importances_)
    }

    return model, X_test, y_test, y_pred, accuracy, cm, report, feature_imp, list(model.classes_)


def predict_career(model, input_values: list):
    """
    Predict career for a single student.

    Parameters
    ----------
    model        : trained DecisionTreeClassifier
    input_values : list of 8 numeric values matching FEATURE_COLS order

    Returns
    -------
    predicted_career : str
    probabilities    : dict {career: probability}
    """
    sample = np.array(input_values).reshape(1, -1)
    predicted_career = model.predict(sample)[0]

    # predict_proba gives confidence for each class
    proba = model.predict_proba(sample)[0]
    probabilities = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(model.classes_, proba)
    }

    return predicted_career, probabilities
