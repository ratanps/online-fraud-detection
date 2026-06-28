-- Online Fraud Transaction Detection — Database Schema
-- Supports ingestion, querying, and analytics for capstone SQL integration

CREATE TABLE IF NOT EXISTS customers (
    customer_id       INTEGER PRIMARY KEY,
    otp_success_rate  REAL,
    past_fraud_count  INTEGER DEFAULT 0,
    past_disputes     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id              INTEGER PRIMARY KEY,
    merchant_category        TEXT NOT NULL,
    historical_fraud_rate    REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS devices (
    device_id         INTEGER PRIMARY KEY,
    device_trust_score REAL
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id              INTEGER PRIMARY KEY,
    customer_id                   INTEGER NOT NULL REFERENCES customers(customer_id),
    device_id                     INTEGER REFERENCES devices(device_id),
    merchant_id                   INTEGER NOT NULL REFERENCES merchants(merchant_id),
    timestamp                     TEXT NOT NULL,
    amount                        REAL NOT NULL,
    payment_method                TEXT NOT NULL,
    is_international              INTEGER DEFAULT 0,
    ip_address_risk_score         REAL,
    txn_count_last_24h            INTEGER,
    avg_amount_last_24h           REAL,
    merchant_diversity_last_7d    INTEGER,
    device_change_flag            INTEGER DEFAULT 0,
    location_change_flag          INTEGER DEFAULT 0,
    authentication_method         TEXT,
    hour_of_day                   INTEGER,
    day_of_week                   INTEGER,
    is_weekend                    INTEGER DEFAULT 0,
    is_fraud                      INTEGER
);

CREATE INDEX idx_txn_customer ON transactions(customer_id);
CREATE INDEX idx_txn_merchant ON transactions(merchant_id);
CREATE INDEX idx_txn_fraud ON transactions(is_fraud);
CREATE INDEX idx_txn_timestamp ON transactions(timestamp);

-- Example analytics queries
-- Fraud rate by payment method:
-- SELECT payment_method, AVG(is_fraud) AS fraud_rate, COUNT(*) AS txn_count
-- FROM transactions GROUP BY payment_method ORDER BY fraud_rate DESC;

-- High-risk transactions in last 24h pattern:
-- SELECT * FROM transactions
-- WHERE ip_address_risk_score > 0.7 AND authentication_method = 'NONE'
-- ORDER BY amount DESC LIMIT 100;
