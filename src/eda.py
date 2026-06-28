"""Exploratory data analysis script for the fraud dataset."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, TARGET_COL  # noqa: E402
from src.data_validation import clean_dataset, validate_dataset  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_eda() -> None:
    df = pd.read_csv(DATA_PATH)
    print("=== Validation ===")
    print(validate_dataset(df))
    df = clean_dataset(df)

    print("\n=== Basic Stats ===")
    print(f"Rows: {len(df):,} | Fraud rate: {df[TARGET_COL].mean():.2%}")
    print(df.describe())

    # Hypothesis: fraud transactions have higher amounts
    fraud_amt = df.loc[df[TARGET_COL] == 1, "amount"]
    legit_amt = df.loc[df[TARGET_COL] == 0, "amount"]
    t_stat, p_value = stats.ttest_ind(fraud_amt, legit_amt, equal_var=False)
    print(f"\n=== T-test: amount (fraud vs legitimate) ===")
    print(f"t-statistic: {t_stat:.4f}, p-value: {p_value:.2e}")

    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.countplot(data=df, x="payment_method", hue=TARGET_COL, ax=axes[0, 0])
    axes[0, 0].set_title("Payment Method vs Fraud")

    sns.boxplot(data=df, x=TARGET_COL, y="amount", ax=axes[0, 1])
    axes[0, 1].set_title("Transaction Amount by Fraud Label")

    fraud_by_cat = df.groupby("merchant_category")[TARGET_COL].mean().sort_values()
    fraud_by_cat.plot(kind="barh", ax=axes[1, 0], color="coral")
    axes[1, 0].set_title("Fraud Rate by Merchant Category")
    axes[1, 0].set_xlabel("Fraud Rate")

    sns.scatterplot(
        data=df.sample(min(5000, len(df)), random_state=42),
        x="ip_address_risk_score",
        y="device_trust_score",
        hue=TARGET_COL,
        alpha=0.5,
        ax=axes[1, 1],
    )
    axes[1, 1].set_title("IP Risk vs Device Trust")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_summary.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved EDA chart to {out_path}")


if __name__ == "__main__":
    run_eda()
