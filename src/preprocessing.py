"""Reusable preprocessing and feature-engineering functions."""

import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Load the raw Telco churn CSV."""
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw Telco data and encode the target."""
    data = df.copy()
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"])
    data = data.drop(columns=["customerID"], errors="ignore")
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    return data

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add reusable engineered features."""
    data = df.copy()
    if "tenure" in data.columns:
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 60, 72],
            labels=["0-1y", "1-2y", "2-4y", "4-5y", "5-6y"]
        )
    return data
