"""Project configuration and feature definitions."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "transactions_fraud_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TARGET_COL = "is_fraud"
ID_COLS = ["transaction_id", "customer_id", "device_id", "merchant_id", "timestamp"]

CATEGORICAL_COLS = [
    "payment_method",
    "merchant_category",
    "authentication_method",
]

NUMERIC_COLS = [
    "amount",
    "is_international",
    "ip_address_risk_score",
    "device_trust_score",
    "txn_count_last_24h",
    "avg_amount_last_24h",
    "merchant_diversity_last_7d",
    "device_change_flag",
    "location_change_flag",
    "otp_success_rate_customer",
    "past_fraud_count_customer",
    "past_disputes_customer",
    "merchant_historical_fraud_rate",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

PAYMENT_METHODS = ["WALLET", "CARD", "UPI", "NETBANKING"]
MERCHANT_CATEGORIES = ["Travel", "Electronics", "Fashion", "Utilities", "Gaming", "Grocery"]
AUTH_METHODS = ["NONE", "OTP", "PIN", "3DS"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

RANDOM_STATE = 42
TEST_SIZE = 0.3
FRAUD_THRESHOLD = 0.5
