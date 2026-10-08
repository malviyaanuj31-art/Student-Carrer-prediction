"""
utils/preprocessing.py
-----------------------
Helper functions for loading and preprocessing the datasets.
These are shared across all model modules.
"""

import os
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")

STUDENT_CSV = os.path.join(DATA_DIR, "student_data.csv")
SALARY_CSV  = os.path.join(DATA_DIR, "salary_data.csv")

# Feature columns used for student classification / clustering
FEATURE_COLS = [
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "PythonSkill",
    "SQLSkill",
    "MLSkill",
    "CommunicationSkill",
    "ProblemSolvingSkill",
]

TARGET_COL = "Career"


def load_student_data():
    """Load the student dataset and return a clean DataFrame."""
    df = pd.read_csv(STUDENT_CSV)

    # Drop rows where any value is missing
    df.dropna(inplace=True)

    # Make sure numeric columns are actually numeric
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)  # drop rows that couldn't be converted
    df.reset_index(drop=True, inplace=True)
    return df


def load_salary_data():
    """Load the salary dataset and return a clean DataFrame."""
    df = pd.read_csv(SALARY_CSV)

    df.dropna(inplace=True)
    df["YearsExperience"] = pd.to_numeric(df["YearsExperience"], errors="coerce")
    df["Salary"]          = pd.to_numeric(df["Salary"],          errors="coerce")
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_feature_matrix(df):
    """Return X (feature matrix) and y (target series) from the student DataFrame."""
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[TARGET_COL].to_numpy()
    return X, y
