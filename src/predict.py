"""Inference and business recommendations for fraud detection."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.config import FRAUD_THRESHOLD, METRICS_PATH, MODEL_PATH, PREPROCESSOR_PATH
from src.preprocessing import input_to_dataframe, load_preprocessor, transform_features


def load_model(path=MODEL_PATH):
    return joblib.load(path)


def load_metrics(path=METRICS_PATH) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}


def predict_fraud(user_input: dict, threshold: float = FRAUD_THRESHOLD) -> dict:
    """Score a transaction and return prediction with recommendations."""
    model = load_model()
    preprocessor = load_preprocessor()
    features = transform_features(input_to_dataframe(user_input), preprocessor)
    proba = float(model.predict_proba(features)[0, 1])
    is_fraud = proba >= threshold

    risk_level = _risk_level(proba)
    recommendations = _recommendations(proba, user_input, is_fraud)

    return {
        "fraud_probability": round(proba, 4),
        "prediction": "Fraud" if is_fraud else "Legitimate",
        "risk_level": risk_level,
        "recommendations": recommendations,
    }


def _risk_level(proba: float) -> str:
    if proba >= 0.75:
        return "Critical"
    if proba >= 0.5:
        return "High"
    if proba >= 0.25:
        return "Medium"
    return "Low"


def _recommendations(proba: float, data: dict, is_fraud: bool) -> list[str]:
    recs = []

    if is_fraud:
        recs.append("Block or hold the transaction pending manual review.")
        recs.append("Notify the fraud investigation team immediately.")
    elif proba >= 0.25:
        recs.append("Approve with enhanced monitoring; flag for post-transaction review.")
    else:
        recs.append("Transaction appears legitimate; proceed with standard processing.")

    if data.get("authentication_method") == "NONE" and proba >= 0.2:
        recs.append("Require step-up authentication (OTP or 3DS) before approval.")

    if data.get("ip_address_risk_score", 0) > 0.7:
        recs.append("High IP risk detected — verify customer identity via callback.")

    if data.get("device_change_flag") or data.get("location_change_flag"):
        recs.append("New device or location detected — send real-time alert to customer.")

    if data.get("past_fraud_count_customer", 0) > 0:
        recs.append("Customer has prior fraud history — apply stricter velocity limits.")

    if data.get("amount", 0) > 15000 and proba >= 0.15:
        recs.append("Large transaction amount — consider additional verification for high-value payments.")

    if data.get("merchant_historical_fraud_rate", 0) > 0.1:
        recs.append("Merchant has elevated historical fraud rate — monitor merchant closely.")

    return recs
