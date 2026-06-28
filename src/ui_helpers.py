"""UI helpers for the Streamlit fraud detection dashboard."""

RISK_COLORS = {
    "Low": "#22c55e",
    "Medium": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
}


def inject_custom_css() -> None:
    import streamlit as st

    st.markdown(
        """
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
        }
        .main-header h1 { color: white !important; margin: 0; font-size: 1.8rem; }
        .main-header p { color: #dbeafe; margin: 0.4rem 0 0 0; }
        .risk-card {
            padding: 1.25rem;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 1rem;
        }
        .signal-chip {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            font-size: 0.85rem;
            margin: 0.2rem;
        }
        .signal-high { background: #fee2e2; color: #991b1b; }
        .signal-med { background: #ffedd5; color: #9a3412; }
        .signal-low { background: #dcfce7; color: #166534; }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            padding: 0.75rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def risk_gauge_html(proba: float, risk_level: str, *, is_safe: bool = False) -> str:
    if is_safe:
        color = RISK_COLORS["Low"]
        label = "Low Risk · Safe"
    else:
        color = RISK_COLORS.get(risk_level, "#64748b")
        label = f"{risk_level} Risk"
    pct = int(proba * 100)
    return f"""
    <div class="risk-card" style="background: {color}18; border: 2px solid {color};">
        <div style="font-size: 2.5rem; font-weight: 700; color: {color};">{pct}%</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: {color};">{label}</div>
        <div style="font-size: 0.9rem; color: #64748b; margin-top: 0.25rem;">Fraud Probability</div>
    </div>
    """


def get_risk_signals(data: dict, proba: float) -> list[tuple[str, str]]:
    """Return human-readable risk signals with severity: high, med, low."""
    signals = []

    if data.get("ip_address_risk_score", 0) >= 0.7:
        signals.append(("High IP address risk", "high"))
    elif data.get("ip_address_risk_score", 0) >= 0.4:
        signals.append(("Elevated IP risk", "med"))

    if data.get("device_trust_score", 1) <= 0.2:
        signals.append(("Very low device trust", "high"))
    elif data.get("device_trust_score", 1) <= 0.4:
        signals.append(("Low device trust", "med"))

    if data.get("authentication_method") == "NONE":
        signals.append(("No authentication used", "high"))

    if data.get("device_change_flag"):
        signals.append(("New device detected", "med"))
    if data.get("location_change_flag"):
        signals.append(("Location change detected", "med"))

    if data.get("past_fraud_count_customer", 0) > 0:
        signals.append(("Customer fraud history", "high"))
    if data.get("past_disputes_customer", 0) >= 3:
        signals.append(("Multiple past disputes", "med"))

    if data.get("txn_count_last_24h", 0) >= 8:
        signals.append(("High transaction velocity (24h)", "med"))

    if data.get("merchant_historical_fraud_rate", 0) >= 0.1:
        signals.append(("High-risk merchant", "high"))

    if data.get("amount", 0) >= 15000:
        signals.append(("Large transaction amount", "med"))

    if proba >= 0.5:
        signals.append(("Model score above fraud threshold", "high"))
    elif proba >= 0.25:
        signals.append(("Model score in watch zone", "med"))
    else:
        signals.append(("Model score within safe range", "low"))

    return signals


def render_signal_chips(signals: list[tuple[str, str]]) -> str:
    css_map = {"high": "signal-high", "med": "signal-med", "low": "signal-low"}
    chips = "".join(
        f'<span class="signal-chip {css_map.get(sev, "signal-med")}">{label}</span>'
        for label, sev in signals
    )
    return f'<div style="line-height: 2;">{chips}</div>'
