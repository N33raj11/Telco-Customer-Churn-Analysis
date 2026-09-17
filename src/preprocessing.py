"""
Telco Customer Churn - preprocessing and feature engineering.

1. Loads the Telco churn CSV.
2. Cleans TotalCharges.
3. Drops customerID.
4. Creates TenureGroup, NumServices and AvgMonthlyCharge.
5. Performs a stratified 80/20 train-test split.
6. Separates numerical and categorical features.
7. Builds a ColumnTransformer with StandardScaler + OneHotEncoder.
8. Saves clean and model-ready train/test CSV files.
9. Saves the fitted preprocessor with joblib.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the source dataset and encode the churn target."""
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Tenure 0 customers have no accumulated charges yet.
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0

    # Defensive fallback for any remaining missing values.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Drop customer identifier before modeling.
    df = df.drop(columns=["customerID"], errors="ignore")

    # Binary target.
    df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business-oriented features."""
    df = df.copy()

    tenure_bins = [-1, 6, 12, 24, 48, np.inf]
    tenure_labels = [
        "0-6 months",
        "7-12 months",
        "13-24 months",
        "25-48 months",
        "49+ months",
    ]

    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
    )

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    # Number of services subscribed to.
    df["NumServices"] = sum(
        (df[col] == "Yes").astype(int) for col in service_cols
    )

    # Average monthly charge based on accumulated charges and tenure.
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    ).round(2)

    return df


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build numeric and categorical preprocessing pipelines."""
    numeric_features = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    return preprocessor, numeric_features, categorical_features


def main() -> None:
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_FILE}\n"
            "Place WA_Fn-UseC_-Telco-Customer-Churn.csv inside data/."
        )

    df = pd.read_csv(RAW_FILE)
    print(f"Original shape: {df.shape}")

    df = clean_data(df)
    df = engineer_features(df)

    # Save cleaned + engineered full dataset.
    df.to_csv(DATA_DIR / "engineered_features.csv", index=False)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Stratified 80/20 split keeps the churn ratio approximately equal.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train.to_csv(DATA_DIR / "clean_X_train.csv", index=False)
    X_test.to_csv(DATA_DIR / "clean_X_test.csv", index=False)
    y_train.to_csv(DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(DATA_DIR / "y_test.csv", index=False)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)

    print("\nNumerical columns:")
    print(numeric_features)

    print("\nCategorical columns:")
    print(categorical_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    ).to_csv(DATA_DIR / "processed_X_train.csv", index=False)

    pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    ).to_csv(DATA_DIR / "processed_X_test.csv", index=False)

    joblib.dump(preprocessor, PROJECT_ROOT / "src" / "preprocessor.joblib")

    print("\n--- Preprocessing complete ---")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Processed train shape: {X_train_processed.shape}")
    print(f"Processed test shape:  {X_test_processed.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate:  {y_test.mean():.2%}")
    print("\nSaved files inside data/ and src/.")


if __name__ == "__main__":
    main()
