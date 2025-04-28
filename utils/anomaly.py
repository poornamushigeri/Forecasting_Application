import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

def detect_anomalies(df, contamination=0.01, random_state=42):
    df = df.copy()

    # Only apply on y-values
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    df['anomaly_score'] = iso.fit_predict(df[['y']])

    # Mark anomalies (score = -1)
    df['is_anomaly'] = df['anomaly_score'] == -1

    return df
