"""Data quality checks for the fraud transaction dataset."""

import pandas as pd

from src.config import CATEGORICAL_COLS, FEATURE_COLS, ID_COLS, NUMERIC_COLS, TARGET_COL


def validate_dataset(df: pd.DataFrame) -> dict:
    """Run validation checks and return a summary report."""
    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "fraud_rate": float(df[TARGET_COL].mean()) if TARGET_COL in df.columns else None,
        "issues": [],
    }

    required = set(FEATURE_COLS + ID_COLS + [TARGET_COL])
    missing_cols = required - set(df.columns)
    if missing_cols:
        report["issues"].append(f"Missing columns: {sorted(missing_cols)}")

    for col in NUMERIC_COLS + [TARGET_COL]:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            report["issues"].append(f"Column '{col}' should be numeric.")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            report["issues"].append(f"Categorical column '{col}' has null values.")

    score_cols = ["ip_address_risk_score", "device_trust_score", "otp_success_rate_customer"]
    for col in score_cols:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range:
                report["issues"].append(f"{col}: {out_of_range} values outside [0, 1].")

    report["is_valid"] = len(report["issues"]) == 0
    return report


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic cleaning steps."""
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates()
    cleaned = cleaned.dropna(subset=FEATURE_COLS + [TARGET_COL])

    for col in ["is_international", "device_change_flag", "location_change_flag", "is_weekend", TARGET_COL]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(int)

    return cleaned.reset_index(drop=True)
