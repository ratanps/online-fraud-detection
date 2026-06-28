"""
Streamlit app — Online Fraud Transaction Detection
Run locally: python -m streamlit run streamlit_app.py
Streamlit Cloud main file path: streamlit_app.py
"""

import json
import sys
from datetime import datetime
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
from src.ui_helpers import (  # noqa: E402
    get_risk_signals,
    inject_custom_css,
    render_signal_chips,
    risk_gauge_html,
)

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_custom_css()

if not (ROOT / "models" / "fraud_model.joblib").exists():
    st.error("Model not found. Train first: `python -m src.train_model`")
    st.stop()

metrics = load_metrics()
best_model = metrics.get("best_model", "Unknown")
best_auc = next((m["roc_auc"] for m in metrics.get("all_models", []) if m["model"] == best_model), 0.0)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def row_to_input(row) -> dict:
    return {
        "amount": float(row["amount"]),
        "payment_method": row["payment_method"],
        "is_international": int(row["is_international"]),
        "merchant_category": row["merchant_category"],
        "ip_address_risk_score": float(row["ip_address_risk_score"]),
        "device_trust_score": float(row["device_trust_score"]),
        "txn_count_last_24h": int(row["txn_count_last_24h"]),
        "avg_amount_last_24h": float(row["avg_amount_last_24h"]),
        "merchant_diversity_last_7d": int(row["merchant_diversity_last_7d"]),
        "device_change_flag": int(row["device_change_flag"]),
        "location_change_flag": int(row["location_change_flag"]),
        "authentication_method": row["authentication_method"],
        "otp_success_rate_customer": float(row["otp_success_rate_customer"]),
        "past_fraud_count_customer": int(row["past_fraud_count_customer"]),
        "past_disputes_customer": int(row["past_disputes_customer"]),
        "merchant_historical_fraud_rate": float(row["merchant_historical_fraud_rate"]),
        "hour_of_day": int(row["hour_of_day"]),
        "day_of_week": int(row["day_of_week"]),
        "is_weekend": int(row["is_weekend"]),
    }


def get_demo_sample(is_fraud: int, exclude_id: int | None = None) -> tuple[dict, int]:
    """Pick a random dataset row with a clear fraud (red) or safe (green) model result."""
    df = load_dataset()
    pool = df[df["is_fraud"] == is_fraud]
    if exclude_id is not None:
        remaining = pool[pool["transaction_id"] != exclude_id]
        if not remaining.empty:
            pool = remaining

    sample_pool = pool.sample(min(200, len(pool)))

    best_input = row_to_input(sample_pool.iloc[0])
    best_id = int(sample_pool.iloc[0]["transaction_id"])
    best_score = -1.0 if is_fraud else 2.0

    for _, row in sample_pool.iterrows():
        data = row_to_input(row)
        proba = predict_fraud(data)["fraud_probability"]
        if is_fraud and proba > best_score:
            best_score = proba
            best_input = data
            best_id = int(row["transaction_id"])
        elif not is_fraud and proba < best_score:
            best_score = proba
            best_input = data
            best_id = int(row["transaction_id"])

    return best_input, best_id


def default_inputs() -> dict:
    return {
        "amount": 5000.0,
        "payment_method": "CARD",
        "is_international": 0,
        "merchant_category": "Electronics",
        "ip_address_risk_score": 0.3,
        "device_trust_score": 0.5,
        "txn_count_last_24h": 2,
        "avg_amount_last_24h": 3000.0,
        "merchant_diversity_last_7d": 3,
        "device_change_flag": 0,
        "location_change_flag": 0,
        "authentication_method": "OTP",
        "otp_success_rate_customer": 0.7,
        "past_fraud_count_customer": 0,
        "past_disputes_customer": 0,
        "merchant_historical_fraud_rate": 0.05,
        "hour_of_day": 12,
        "day_of_week": 2,
        "is_weekend": 0,
    }


if "inputs" not in st.session_state:
    st.session_state.inputs = default_inputs()
if "history" not in st.session_state:
    st.session_state.history = []
if "show_prediction" not in st.session_state:
    st.session_state.show_prediction = False
if "form_key" not in st.session_state:
    st.session_state.form_key = 0
if "last_fraud_id" not in st.session_state:
    st.session_state.last_fraud_id = None
if "last_safe_id" not in st.session_state:
    st.session_state.last_safe_id = None


def _load_fraud_sample() -> None:
    inputs, txn_id = get_demo_sample(1, exclude_id=st.session_state.last_fraud_id)
    st.session_state.inputs = inputs
    st.session_state.last_fraud_id = txn_id
    st.session_state.show_prediction = True
    st.session_state.form_key += 1


def _load_safe_sample() -> None:
    inputs, txn_id = get_demo_sample(0, exclude_id=st.session_state.last_safe_id)
    st.session_state.inputs = inputs
    st.session_state.last_safe_id = txn_id
    st.session_state.show_prediction = True
    st.session_state.form_key += 1


def _reset_form() -> None:
    st.session_state.inputs = default_inputs()
    st.session_state.show_prediction = False
    st.session_state.last_fraud_id = None
    st.session_state.last_safe_id = None
    st.session_state.form_key += 1


def _run_detect() -> None:
    st.session_state.show_prediction = True


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=72)
    st.title("Control Panel")
    st.caption("Adjust settings and view model info")

    threshold = st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.05, help="Transactions above this score are flagged as fraud")

    st.divider()
    st.markdown("**Model Performance**")
    st.metric("Best Model", best_model)
    st.metric("ROC-AUC", f"{best_auc:.3f}" if best_auc else "N/A")

    df_sidebar = load_dataset()
    st.metric("Dataset Size", f"{len(df_sidebar):,}")
    st.metric("Fraud Rate", f"{df_sidebar['is_fraud'].mean():.1%}")

    st.divider()
    if st.session_state.history:
        st.markdown("**Recent Checks**")
        for item in reversed(st.session_state.history[-5:]):
            icon = "🔴" if item["prediction"] == "Fraud" else "🟢"
            st.caption(f"{icon} {item['prediction']} — {item['proba']:.0%} ({item['time']})")


# --- Header ---
st.markdown(
    """
    <div class="main-header">
        <h1>🛡️ Online Fraud Transaction Detection</h1>
        <p>Real-time ML scoring · Risk assessment · Actionable recommendations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_predict, tab_eda, tab_about = st.tabs(["🔍 Fraud Check", "📊 Data Overview", "📋 Project Info"])


def render_inputs() -> dict:
    """Render input widgets and return user_input dict."""
    fk = st.session_state.form_key
    s = st.session_state.inputs

    btn1, btn2, btn3 = st.columns(3)
    with btn1:
        st.button("🚨 Load fraud sample", width="stretch", on_click=_load_fraud_sample)
    with btn2:
        st.button("✅ Load safe sample", width="stretch", on_click=_load_safe_sample)
    with btn3:
        st.button("🔄 Reset form", width="stretch", on_click=_reset_form)

    st.button("🔍 Detect Fraud", type="primary", width="stretch", on_click=_run_detect)

    st.info(
        "You can change the values below and click **Detect Fraud** to see the updated result. "
        "Sample buttons load a new random example each time you click them."
    )

    with st.expander("💳 Transaction Details", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            amount = st.number_input("Amount ($)", min_value=0.0, value=float(s.get("amount", 5000.0)), key=f"amount_{fk}")
            payment_method = st.selectbox(
                "Payment Method",
                PAYMENT_METHODS,
                index=PAYMENT_METHODS.index(s.get("payment_method", "CARD")) if s.get("payment_method") in PAYMENT_METHODS else 1,
                key=f"payment_method_{fk}",
            )
            merchant_category = st.selectbox(
                "Merchant Category",
                MERCHANT_CATEGORIES,
                index=MERCHANT_CATEGORIES.index(s.get("merchant_category", "Electronics")) if s.get("merchant_category") in MERCHANT_CATEGORIES else 1,
                key=f"merchant_category_{fk}",
            )
            auth_method = st.selectbox(
                "Authentication",
                AUTH_METHODS,
                index=AUTH_METHODS.index(s.get("authentication_method", "OTP")) if s.get("authentication_method") in AUTH_METHODS else 1,
                key=f"auth_method_{fk}",
            )
        with c2:
            is_international = st.checkbox("International Transaction", value=bool(s.get("is_international", 0)), key=f"intl_{fk}")
            device_change = st.checkbox("Device Change Flag", value=bool(s.get("device_change_flag", 0)), key=f"dev_chg_{fk}")
            location_change = st.checkbox("Location Change Flag", value=bool(s.get("location_change_flag", 0)), key=f"loc_chg_{fk}")
            is_weekend = st.checkbox("Weekend", value=bool(s.get("is_weekend", 0)), key=f"weekend_{fk}")

    with st.expander("⚠️ Risk Signals", expanded=True):
        c3, c4 = st.columns(2)
        with c3:
            ip_risk = st.slider("IP Address Risk Score", 0.0, 1.0, float(s.get("ip_address_risk_score", 0.3)), key=f"ip_risk_{fk}")
            device_trust = st.slider("Device Trust Score", 0.0, 1.0, float(s.get("device_trust_score", 0.5)), key=f"dev_trust_{fk}")
            otp_rate = st.slider("Customer OTP Success Rate", 0.0, 1.0, float(s.get("otp_success_rate_customer", 0.7)), key=f"otp_rate_{fk}")
            merchant_fraud_rate = st.slider("Merchant Historical Fraud Rate", 0.0, 1.0, float(s.get("merchant_historical_fraud_rate", 0.05)), key=f"merch_fraud_{fk}")
        with c4:
            txn_count_24h = st.number_input("Transactions (last 24h)", min_value=0, value=int(s.get("txn_count_last_24h", 2)), key=f"txn_24h_{fk}")
            avg_amount_24h = st.number_input("Avg Amount (last 24h)", min_value=0.0, value=float(s.get("avg_amount_last_24h", 3000.0)), key=f"avg_24h_{fk}")
            merchant_diversity = st.number_input("Merchant Diversity (7d)", min_value=0, value=int(s.get("merchant_diversity_last_7d", 3)), key=f"merch_div_{fk}")
            past_fraud = st.number_input("Past Fraud Count (customer)", min_value=0, value=int(s.get("past_fraud_count_customer", 0)), key=f"past_fraud_{fk}")
            past_disputes = st.number_input("Past Disputes (customer)", min_value=0, value=int(s.get("past_disputes_customer", 0)), key=f"past_disp_{fk}")

    with st.expander("🕐 Time Context", expanded=False):
        hour = st.slider("Hour of Day", 0, 23, int(s.get("hour_of_day", 12)), key=f"hour_{fk}")
        day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: DAY_NAMES[x], index=int(s.get("day_of_week", 2)), key=f"day_{fk}")

    return {
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


def render_result(user_input: dict) -> None:
    result = predict_fraud(user_input, threshold=threshold)
    proba = result["fraud_probability"]
    is_fraud = result["prediction"] == "Fraud"
    is_safe = not is_fraud and proba < 0.25
    signals = get_risk_signals(user_input, proba)

    st.markdown(risk_gauge_html(proba, result["risk_level"], is_safe=is_safe), unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Decision", result["prediction"])
    r2.metric("Amount", f"${user_input['amount']:,.2f}")
    r3.metric("Payment", user_input["payment_method"])

    if is_fraud:
        st.error(f"⚠️ **FRAUD DETECTED** — Block or escalate this transaction")
    elif is_safe:
        st.success(f"✅ **LEGITIMATE · SAFE** — Transaction approved with low fraud risk")
    else:
        st.warning(f"👀 **REVIEW NEEDED** — Medium risk; enhanced monitoring recommended")

    st.progress(min(proba, 1.0))

    st.markdown("**Key Risk Signals**")
    st.markdown(render_signal_chips(signals), unsafe_allow_html=True)

    st.markdown("**Recommended Actions**")
    for i, rec in enumerate(result["recommendations"], 1):
        st.info(f"{i}. {rec}")

    with st.expander("📄 Transaction summary"):
        summary = pd.DataFrame(
            [{"Field": k, "Value": str(v)} for k, v in user_input.items()]
        )
        st.dataframe(summary, width="stretch", hide_index=True)

    if st.button("💾 Save to history", width="stretch"):
        st.session_state.history.append(
            {
                "prediction": result["prediction"],
                "proba": proba,
                "time": datetime.now().strftime("%H:%M:%S"),
                "amount": user_input["amount"],
            }
        )
        st.toast("Saved to sidebar history!", icon="✅")

    st.download_button(
        "⬇️ Download report (JSON)",
        data=json.dumps({"input": user_input, "result": result}, indent=2),
        file_name="fraud_check_report.json",
        mime="application/json",
        width="stretch",
    )


EMPTY_RESULT_MSG = """
👈 **No prediction yet**

1. Enter transaction details below, or load a sample  
2. Click **Detect Fraud** to see the result  

You can edit any field after loading a sample and click **Detect Fraud** again to refresh the prediction.
"""


with tab_predict:
    col_form, col_result = st.columns([1.3, 1])

    with col_form:
        st.subheader("Transaction Attributes")
        user_input = render_inputs()

    with col_result:
        st.subheader("Prediction Result")
        if st.session_state.show_prediction:
            render_result(user_input)
        else:
            st.info(EMPTY_RESULT_MSG)


with tab_eda:
    st.subheader("Dataset Overview")
    df = load_dataset()

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Transactions", f"{len(df):,}")
    m2.metric("Fraud Cases", f"{df['is_fraud'].sum():,}")
    m3.metric("Fraud Rate", f"{df['is_fraud'].mean():.2%}")
    m4.metric("Features", len(df.columns) - 1)
    m5.metric("Best Model", best_model)

    chart1, chart2, chart3 = st.columns(3)

    with chart1:
        st.markdown("**Fraud by Payment Method**")
        fraud_by_payment = df.groupby("payment_method")["is_fraud"].mean().sort_values(ascending=False)
        st.bar_chart(fraud_by_payment, color="#ef4444")

    with chart2:
        st.markdown("**Fraud by Merchant Category**")
        fraud_by_merchant = df.groupby("merchant_category")["is_fraud"].mean().sort_values(ascending=False)
        st.bar_chart(fraud_by_merchant, color="#f97316")

    with chart3:
        st.markdown("**Fraud vs Legitimate Split**")
        split = df["is_fraud"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
        st.bar_chart(split, color="#2563eb")

    st.markdown("**Amount Distribution by Fraud Label**")
    st.scatter_chart(
        df.sample(2000, random_state=42),
        x="amount",
        y="ip_address_risk_score",
        color="is_fraud",
        size="txn_count_last_24h",
    )

    with st.expander("View sample records"):
        st.dataframe(df.head(20), width="stretch")

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
        st.dataframe(
            comparison.style.highlight_max(subset=["ROC-AUC", "F1", "Recall"], color="#dcfce7"),
            width="stretch",
        )


with tab_about:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            ### Business Problem
            Online payment platforms lose billions annually to fraud.
            This system scores transactions in real time and recommends
            **approve**, **step-up auth**, or **block**.

            ### Capstone Requirements Met
            | Requirement | Status |
            |-------------|--------|
            | User input & prediction | ✅ |
            | Data validation & cleaning | ✅ |
            | EDA & statistical analysis | ✅ |
            | Model training & metrics | ✅ |
            | Strategic recommendations | ✅ |
            | Interactive dashboard | ✅ |
            | SQL schema | ✅ |
            """
        )
    with c2:
        st.markdown(
            """
            ### Methodology
            1. Data validation & cleaning
            2. Feature scaling + one-hot encoding
            3. Logistic Regression, Random Forest, XGBoost
            4. ROC-AUC, F1, precision, recall evaluation
            5. Streamlit deployment with on-demand fraud scoring

            ### How to Use This Dashboard
            1. Open **Fraud Check** tab
            2. Enter transaction attributes or load a sample
            3. Click **Detect Fraud** to view results
            4. Explore **Data Overview** for EDA insights
            """
        )

    if METRICS_PATH.exists():
        with st.expander("Full model metrics (JSON)"):
            st.code(json.dumps(metrics, indent=2), language="json")
