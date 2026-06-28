"""
Streamlit app — Online Fraud Transaction Detection
Run locally: python -m streamlit run streamlit_app.py
Streamlit Cloud main file path: streamlit_app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    AUTH_METHODS,
    DATA_PATH,
    DAY_NAMES,
    MERCHANT_CATEGORIES,
    METRICS_PATH,
    PAYMENT_METHODS,
)
from src.predict import load_metrics, predict_fraud  # noqa: E402

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Online Fraud Transaction Detection")
st.caption("Capstone project — ML-powered real-time fraud scoring and recommendations")

if not (ROOT / "models" / "fraud_model.joblib").exists():
    st.error(
        "Model not found. Train the model first:\n\n"
        "```\npython -m src.train_model\n```"
    )
    st.stop()

metrics = load_metrics()
best_model = metrics.get("best_model", "Unknown")

tab_predict, tab_eda, tab_about = st.tabs(["🔍 Fraud Check", "📊 Data Overview", "📋 Project Info"])


def default_sample() -> dict:
    """Load a random fraud and legitimate sample from the dataset."""
    df = pd.read_csv(DATA_PATH)
    fraud_row = df[df["is_fraud"] == 1].sample(1, random_state=42).iloc[0]
    return {
        "amount": float(fraud_row["amount"]),
        "payment_method": fraud_row["payment_method"],
        "is_international": int(fraud_row["is_international"]),
        "merchant_category": fraud_row["merchant_category"],
        "ip_address_risk_score": float(fraud_row["ip_address_risk_score"]),
        "device_trust_score": float(fraud_row["device_trust_score"]),
        "txn_count_last_24h": int(fraud_row["txn_count_last_24h"]),
        "avg_amount_last_24h": float(fraud_row["avg_amount_last_24h"]),
        "merchant_diversity_last_7d": int(fraud_row["merchant_diversity_last_7d"]),
        "device_change_flag": int(fraud_row["device_change_flag"]),
        "location_change_flag": int(fraud_row["location_change_flag"]),
        "authentication_method": fraud_row["authentication_method"],
        "otp_success_rate_customer": float(fraud_row["otp_success_rate_customer"]),
        "past_fraud_count_customer": int(fraud_row["past_fraud_count_customer"]),
        "past_disputes_customer": int(fraud_row["past_disputes_customer"]),
        "merchant_historical_fraud_rate": float(fraud_row["merchant_historical_fraud_rate"]),
        "hour_of_day": int(fraud_row["hour_of_day"]),
        "day_of_week": int(fraud_row["day_of_week"]),
        "is_weekend": int(fraud_row["is_weekend"]),
    }


with tab_predict:
    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.subheader("Transaction Attributes")
        if st.button("Load sample (known fraud case)"):
            st.session_state["sample"] = default_sample()

        sample = st.session_state.get("sample", {})

        with st.form("fraud_form"):
            c1, c2 = st.columns(2)

            with c1:
                amount = st.number_input("Amount ($)", min_value=0.0, value=sample.get("amount", 5000.0))
                payment_method = st.selectbox("Payment Method", PAYMENT_METHODS, index=PAYMENT_METHODS.index(sample.get("payment_method", "CARD")) if sample.get("payment_method") in PAYMENT_METHODS else 1)
                merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES, index=MERCHANT_CATEGORIES.index(sample.get("merchant_category", "Electronics")) if sample.get("merchant_category") in MERCHANT_CATEGORIES else 1)
                auth_method = st.selectbox("Authentication", AUTH_METHODS, index=AUTH_METHODS.index(sample.get("authentication_method", "OTP")) if sample.get("authentication_method") in AUTH_METHODS else 1)
                is_international = st.checkbox("International Transaction", value=bool(sample.get("is_international", 0)))
                device_change = st.checkbox("Device Change Flag", value=bool(sample.get("device_change_flag", 0)))
                location_change = st.checkbox("Location Change Flag", value=bool(sample.get("location_change_flag", 0)))

            with c2:
                ip_risk = st.slider("IP Address Risk Score", 0.0, 1.0, float(sample.get("ip_address_risk_score", 0.3)))
                device_trust = st.slider("Device Trust Score", 0.0, 1.0, float(sample.get("device_trust_score", 0.5)))
                otp_rate = st.slider("Customer OTP Success Rate", 0.0, 1.0, float(sample.get("otp_success_rate_customer", 0.7)))
                merchant_fraud_rate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, float(sample.get("merchant_historical_fraud_rate", 0.05)))
                txn_count_24h = st.number_input("Transactions (last 24h)", min_value=0, value=int(sample.get("txn_count_last_24h", 2)))
                avg_amount_24h = st.number_input("Avg Amount (last 24h)", min_value=0.0, value=float(sample.get("avg_amount_last_24h", 3000.0)))
                merchant_diversity = st.number_input("Merchant Diversity (7d)", min_value=0, value=int(sample.get("merchant_diversity_last_7d", 3)))
                past_fraud = st.number_input("Past Fraud Count (customer)", min_value=0, value=int(sample.get("past_fraud_count_customer", 0)))
                past_disputes = st.number_input("Past Disputes (customer)", min_value=0, value=int(sample.get("past_disputes_customer", 0)))
                hour = st.slider("Hour of Day", 0, 23, int(sample.get("hour_of_day", 12)))
                day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: DAY_NAMES[x], index=int(sample.get("day_of_week", 2)))
                is_weekend = st.checkbox("Weekend", value=bool(sample.get("is_weekend", 0)))

            submitted = st.form_submit_button("Detect Fraud", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")
        if submitted:
            user_input = {
                "amount": amount,
                "payment_method": payment_method,
                "is_international": int(is_international),
                "merchant_category": merchant_category,
                "ip_address_risk_score": ip_risk,
                "device_trust_score": device_trust,
                "txn_count_last_24h": txn_count_24h,
                "avg_amount_last_24h": avg_amount_24h,
                "merchant_diversity_last_7d": merchant_diversity,
                "device_change_flag": int(device_change),
                "location_change_flag": int(location_change),
                "authentication_method": auth_method,
                "otp_success_rate_customer": otp_rate,
                "past_fraud_count_customer": past_fraud,
                "past_disputes_customer": past_disputes,
                "merchant_historical_fraud_rate": merchant_fraud_rate,
                "hour_of_day": hour,
                "day_of_week": day,
                "is_weekend": int(is_weekend),
            }

            result = predict_fraud(user_input)
            proba = result["fraud_probability"]
            is_fraud = result["prediction"] == "Fraud"

            st.metric("Fraud Probability", f"{proba:.1%}")
            st.metric("Risk Level", result["risk_level"])

            if is_fraud:
                st.error(f"⚠️ **{result['prediction']}** — Transaction flagged as suspicious")
            else:
                st.success(f"✅ **{result['prediction']}** — Transaction appears safe")

            st.progress(min(proba, 1.0))

            st.markdown("**Recommended Actions**")
            for rec in result["recommendations"]:
                st.info(rec)
        else:
            st.markdown(
                "Enter transaction details on the left and click **Detect Fraud** "
                "to get a prediction and business recommendations."
            )

with tab_eda:
    st.subheader("Dataset Overview")
    df = pd.read_csv(DATA_PATH)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Transactions", f"{len(df):,}")
    m2.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
    m3.metric("Features", len(df.columns) - 1)
    m4.metric("Best Model", best_model)

    st.markdown("**Fraud Rate by Payment Method**")
    fraud_by_payment = df.groupby("payment_method")["is_fraud"].mean().sort_values(ascending=False)
    st.bar_chart(fraud_by_payment)

    st.markdown("**Fraud Rate by Merchant Category**")
    fraud_by_merchant = df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False)
    st.bar_chart(fraud_by_merchant)

    with st.expander("View sample records"):
        st.dataframe(df.head(20), use_container_width=True)

    if metrics.get("all_models"):
        st.markdown("**Model Comparison (Test Set)**")
        comparison = pd.DataFrame(
            [
                {
                    "Model": m["model"],
                    "ROC-AUC": m["roc_auc"],
                    "F1": m["f1"],
                    "Precision": m["precision"],
                    "Recall": m["recall"],
                }
                for m in metrics["all_models"]
            ]
        )
        st.dataframe(comparison.style.format({"ROC-AUC": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}"}), use_container_width=True)

with tab_about:
    st.markdown(
        """
        ### Business Problem
        Online payment platforms lose billions annually to fraudulent transactions.
        This project builds an ML system to **score transactions in real time** and
        recommend appropriate actions (approve, step-up auth, or block).

        ### Analytical Objectives
        - Achieve strong fraud detection performance (ROC-AUC, F1, recall)
        - Identify high-risk transaction patterns (payment method, auth, device signals)
        - Deploy an interactive dashboard for fraud analysts

        ### Methodology
        1. **Data validation & cleaning** — duplicates, missing values, type checks
        2. **Feature engineering** — numeric scaling + one-hot encoding for categoricals
        3. **Model training** — Logistic Regression, Random Forest, XGBoost
        4. **Evaluation** — ROC-AUC, precision, recall, F1, confusion matrix
        5. **Deployment** — Streamlit app with rule-based recommendations

        ### Dataset
        `transactions_fraud_dataset.csv` — 50,000 synthetic online payment transactions
        with customer, device, merchant, and behavioral features.

        ### SQL Integration
        See `sql/schema.sql` for normalized schema and sample analytics queries.

        ### How to Run
        ```bash
        pip install -r requirements.txt
        python -m src.train_model
        streamlit run streamlit_app.py
        ```
        """
    )

    if METRICS_PATH.exists():
        with st.expander("Full model metrics (JSON)"):
            st.code(json.dumps(metrics, indent=2), language="json")
