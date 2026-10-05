"""General project helpers."""

def validate_columns(df, required_columns):
    """Raise an error when required columns are missing."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True
