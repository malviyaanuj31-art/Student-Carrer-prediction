"""
models/clustering.py
---------------------
K-Means Clustering – groups students into performance clusters.

Demonstrates:
  - Unsupervised learning
  - K-Means with k=3
  - Cluster visualization
  - Cluster labelling based on average marks
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from utils.preprocessing import load_student_data

# Features used for clustering (subset of all features)
CLUSTER_FEATURES = [
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "PythonSkill",
    "MLSkill",
]

N_CLUSTERS = 3


def run_clustering():
    """
    Run K-Means clustering on the student dataset.

    Returns
    -------
    df_result  : DataFrame with original data + 'Cluster' and 'ClusterLabel' columns
    kmeans     : fitted KMeans object
    scaler     : fitted StandardScaler
    summary    : DataFrame with per-cluster averages
    """
    df = load_student_data()

    X = df[CLUSTER_FEATURES].to_numpy(dtype=float)

    # Scale features so that no single feature dominates the distance metric
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means with 3 clusters, multiple random initialisations for stability
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    df = df.copy()
    df["Cluster"] = kmeans.fit_predict(X_scaled)

    # Assign human-readable labels based on average PreviousMarks per cluster
    # The cluster with the highest average marks → "Strong"
    # The lowest → "Needs Improvement", middle → "Average"
    avg_marks = df.groupby("Cluster")["PreviousMarks"].mean()
    sorted_clusters = avg_marks.sort_values().index.tolist()

    label_map = {
        sorted_clusters[0]: "Needs Improvement",
        sorted_clusters[1]: "Average",
        sorted_clusters[2]: "Strong",
    }
    df["ClusterLabel"] = df["Cluster"].map(label_map)

    # Per-cluster summary
    summary = (
        df.groupby("ClusterLabel")[CLUSTER_FEATURES]
        .mean()
        .round(2)
        .reset_index()
    )

    return df, kmeans, scaler, summary, label_map


def predict_cluster(kmeans, scaler, label_map, input_values: list):
    """
    Predict which cluster a new student belongs to.

    Parameters
    ----------
    input_values : list of 5 values matching CLUSTER_FEATURES order

    Returns
    -------
    cluster_id    : int
    cluster_label : str
    """
    sample = np.array(input_values).reshape(1, -1)
    sample_scaled = scaler.transform(sample)
    cluster_id = int(kmeans.predict(sample_scaled)[0])
    cluster_label = label_map.get(cluster_id, f"Cluster {cluster_id}")
    return cluster_id, cluster_label
