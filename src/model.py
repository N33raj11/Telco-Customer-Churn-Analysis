"""Model training and persistence helpers."""

from pathlib import Path
import joblib

def save_model(model, path: str) -> None:
    """Save a fitted model/pipeline."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    """Load a saved model/pipeline."""
    return joblib.load(path)


