"""Train and evaluate fraud detection models."""

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATA_PATH,
    FEATURE_COLS,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from src.data_validation import clean_dataset, validate_dataset  # noqa: E402
from src.preprocessing import fit_preprocessor, save_preprocessor, transform_features  # noqa: E402


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    report = validate_dataset(df)
    print("Validation report:", json.dumps(report, indent=2))
    if not report["is_valid"]:
        print("Warning: validation issues found; proceeding after cleaning.")
    return clean_dataset(df)


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def train_models(df: pd.DataFrame) -> tuple:
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = fit_preprocessor(X_train)
    X_train_t = transform_features(X_train, preprocessor)
    X_test_t = transform_features(X_test, preprocessor)

    fraud_ratio = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, criterion="entropy", class_weight="balanced", random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=fraud_ratio,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }

    results = []
    best_name, best_model, best_auc = None, None, -1.0

    for name, model in candidates.items():
        model.fit(X_train_t, y_train)
        metrics = evaluate_model(name, model, X_test_t, y_test)
        results.append(metrics)
        print(f"\n{name} — ROC-AUC: {metrics['roc_auc']:.4f}, F1: {metrics['f1']:.4f}")
        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_model = model

    return best_name, best_model, preprocessor, results, {"X_test": X_test, "y_test": y_test}


def save_artifacts(model, preprocessor, results, best_name: str) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_preprocessor(preprocessor)

    payload = {
        "best_model": best_name,
        "model_path": str(MODEL_PATH),
        "all_models": results,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved best model ({best_name}) to {MODEL_PATH}")


def main() -> None:
    df = load_data()
    print(f"Training on {len(df):,} transactions | Fraud rate: {df[TARGET_COL].mean():.2%}")
    best_name, best_model, preprocessor, results, _ = train_models(df)
    save_artifacts(best_model, preprocessor, results, best_name)


if __name__ == "__main__":
    main()
