"""
models/regression.py
---------------------
Linear Regression for Salary Prediction.

Demonstrates:
  - Simple linear regression
  - Scatter plot + regression line
  - Evaluation metrics: MAE, MSE, RMSE, R²
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.preprocessing import load_salary_data


def train_regression():
    """
    Train a Linear Regression model on the salary dataset.

    Returns
    -------
    model    : trained LinearRegression
    df       : the full salary DataFrame (for plotting)
    metrics  : dict with MAE, MSE, RMSE, R2
    X_test   : test X values
    y_test   : true test salaries
    y_pred   : predicted test salaries
    """
    df = load_salary_data()

    X = df[["YearsExperience"]].to_numpy(dtype=float)
    y = df["Salary"].to_numpy(dtype=float)

    # 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae  = float(mean_absolute_error(y_test, y_pred))
    mse  = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(y_test, y_pred))

    metrics = {
        "MAE":  round(mae,  2),
        "MSE":  round(mse,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
    }

    return model, df, metrics, X_test, y_test, y_pred


def predict_salary(model, years_experience: float):
    """
    Predict salary for a given number of years of experience.

    Returns predicted salary as a float.
    """
    sample = np.array([[years_experience]])
    predicted = model.predict(sample)[0]
    return round(float(predicted), 2)
