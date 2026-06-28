# Online Fraud Transaction Detection — Capstone Project

ML-powered fraud detection system for online payment transactions, built as a Data Science capstone submission. Includes data validation, EDA, model training, SQL schema, and a **Streamlit deployment dashboard**.

## Business Problem

Payment platforms need to identify fraudulent transactions in real time while minimizing false positives. This project trains supervised ML models on transaction, customer, device, and merchant signals to **predict fraud** and **recommend actions** (approve, step-up authentication, or block).

## Project Structure

```
├── transactions_fraud_dataset.csv   # 50K transaction records
├── app/
│   └── streamlit_app.py             # Deployable fraud detection dashboard
├── src/
│   ├── config.py                    # Paths, feature definitions
│   ├── data_validation.py           # Data quality checks & cleaning
│   ├── preprocessing.py             # Feature pipeline (scale + encode)
│   ├── eda.py                       # Exploratory analysis & hypothesis test
│   ├── train_model.py               # Train & compare ML models
│   └── predict.py                   # Inference + business recommendations
├── sql/
│   └── schema.sql                   # Database schema + sample queries
├── models/                          # Saved model artifacts (after training)
├── reports/figures/                 # EDA output charts
└── requirements.txt
```

## Dataset

| Attribute | Description |
|-----------|-------------|
| `amount` | Transaction amount |
| `payment_method` | WALLET, CARD, UPI, NETBANKING |
| `merchant_category` | Travel, Electronics, Fashion, etc. |
| `ip_address_risk_score` | IP risk (0–1) |
| `device_trust_score` | Device trust (0–1) |
| `authentication_method` | NONE, OTP, PIN, 3DS |
| `past_fraud_count_customer` | Customer fraud history |
| `is_fraud` | Target label (0/1) |

Full schema: 25 columns including behavioral features (txn velocity, device/location flags, merchant fraud rate).

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.train_model
```

This validates data, trains Logistic Regression, Random Forest, and XGBoost, saves the best model to `models/fraud_model.joblib`, and writes metrics to `models/metrics.json`.

### 3. Run EDA (optional)

```bash
python -m src.eda
```

Generates `reports/figures/eda_summary.png` and prints a t-test on transaction amounts.

### 4. Launch Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

## Streamlit Dashboard

The app has three tabs:

- **Fraud Check** — Enter transaction attributes; get fraud probability, risk level, and recommendations
- **Data Overview** — Dataset stats, fraud rates by category, model comparison
- **Project Info** — Methodology and capstone alignment

Use **Load sample (known fraud case)** to pre-fill inputs from the dataset.

## Capstone Alignment

| Requirement | Implementation |
|-------------|----------------|
| Business problem & objectives | Documented in app + README |
| Data validation & cleaning | `src/data_validation.py` |
| Database schema | `sql/schema.sql` |
| Python & EDA | `src/eda.py` |
| Statistical analysis | T-test on fraud vs legitimate amounts |
| Model training & testing | `src/train_model.py` (ROC-AUC, F1, precision, recall) |
| Strategic recommendations | `src/predict.py` rule-based actions |
| Code modularity | Separate modules under `src/` |
| Dashboard (bonus) | Streamlit app with UX-focused layout |

## Models

Three classifiers are compared on a 70/30 stratified split:

- Logistic Regression (class-balanced)
- Random Forest (100 trees, entropy)
- XGBoost (scale_pos_weight for imbalance)

Best model by ROC-AUC is saved for production inference.

## Reference

Inspired by the sample notebook `Online_Payment_Fraud_Detection_using_Machine_Learning_in_Python.ipynb`, adapted for the custom `transactions_fraud_dataset.csv` schema.

## Author

Data Science Capstone Project
