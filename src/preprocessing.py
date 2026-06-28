"""Feature preprocessing pipeline for fraud detection."""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_COLS,
    PREPROCESSOR_PATH,
)


def build_preprocessor() -> ColumnTransformer:
    """Create sklearn preprocessor for numeric scaling and categorical encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        remainder="drop",
    )


def fit_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Fit preprocessor on training features."""
    preprocessor = build_preprocessor()
    preprocessor.fit(df[FEATURE_COLS])
    return preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, path=PREPROCESSOR_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path=PREPROCESSOR_PATH) -> ColumnTransformer:
    return joblib.load(path)


def transform_features(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    """Transform raw input dataframe into model-ready features."""
    return preprocessor.transform(df[FEATURE_COLS])


def input_to_dataframe(user_input: dict) -> pd.DataFrame:
    """Convert a single transaction dict into a one-row dataframe."""
    return pd.DataFrame([{col: user_input[col] for col in FEATURE_COLS}])
