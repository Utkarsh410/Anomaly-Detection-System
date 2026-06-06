# Anomaly Detection System

A **production-style real-time fraud detection system** that scores incoming financial transactions using an ensemble of supervised and unsupervised machine-learning models, stores results in PostgreSQL, and exposes the scoring pipeline via a FastAPI REST API.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Dataset](#4-dataset)
5. [Machine Learning Models](#5-machine-learning-models)
6. [Ensemble Strategy](#6-ensemble-strategy)
7. [Risk Scoring](#7-risk-scoring)
8. [Alert System](#8-alert-system)
9. [Database Design](#9-database-design)
10. [API Reference](#10-api-reference)
11. [Training Pipeline](#11-training-pipeline)
12. [Model Performance](#12-model-performance)
13. [Getting Started](#13-getting-started)
14. [Docker Deployment](#14-docker-deployment)
15. [Challenges & Solutions](#15-challenges--solutions)
16. [Future Improvements](#16-future-improvements)

---

## 1. Project Overview

This project implements a **real-time anomaly detection system** for identifying fraudulent financial transactions. The solution follows a production-grade fintech architecture that combines:

- **Hybrid ML models** — supervised XGBoost + unsupervised Isolation Forest
- **Ensemble scoring** — weighted combination to improve generalization
- **Risk-based alerting** — four-tier classification (CRITICAL / HIGH / MEDIUM / LOW)
- **Database audit trail** — PostgreSQL via SQLAlchemy ORM
- **REST API** — FastAPI with automatic OpenAPI docs at `/docs`
- **Containerized deployment** — Docker + docker-compose
- **Transferable architecture** — the same ensemble + risk-tier + 
  audit-trail pattern applies directly to review fraud detection 
  (fake reviews, seller manipulation), network intrusion detection, 
  and trading surveillance — anywhere behaviour needs to be scored 
  against a highly imbalanced anomaly baseline
  
---

## 2. Architecture

```
POST /detect
      │
      ▼
TransactionInput validation (Pydantic)
      │
      ▼
predict_score()
  ├── XGBoost  → fraud probability (0–1)         × 0.70
  └── Isolation Forest → anomaly flag (0 or 1)   × 0.30
      │
      ▼
get_risk_level()   →   CRITICAL | HIGH | MEDIUM | LOW
      │
      ▼
generate_alert()   →   alert payload
      │
      ▼
PostgreSQL (transactions table)
      │
      ▼
JSON response  { transaction_id, score, risk, alert }
```

---

## 3. Tech Stack

| Layer           | Technology                          |
|-----------------|-------------------------------------|
| API framework   | FastAPI + Uvicorn                   |
| ML – supervised | XGBoost                             |
| ML – unsupervised | scikit-learn Isolation Forest     |
| Preprocessing   | scikit-learn StandardScaler         |
| Serialisation   | joblib                              |
| Database        | PostgreSQL 15                       |
| ORM             | SQLAlchemy 2.x                      |
| Validation      | Pydantic v2                         |
| Containerisation| Docker + docker-compose             |

---

## 4. Dataset

**Credit Card Fraud Detection Dataset** (Kaggle / ULB Machine Learning Group)

| Feature  | Description                            |
|----------|----------------------------------------|
| Time     | Seconds elapsed since first transaction |
| V1–V28   | PCA-transformed anonymised features   |
| Amount   | Transaction amount (USD)               |
| Class    | Target label — 0 = normal, 1 = fraud  |

- 284,807 total transactions
- ~0.17% fraud rate (highly imbalanced)
- `creditcard.csv` is not included in this repository; download from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root before training.

---
### 4.1 Key EDA Findings

- **V14, V10, V12** show the strongest separation between fraud and 
  normal classes — consistent with literature on this dataset
- **Fraud amounts skew lower** (median ~$9) than normal transactions 
  (median ~$22), suggesting card-testing behaviour with small 
  probe transactions before larger withdrawals
- **Temporal clustering:** fraud concentrates in off-peak hours 
  (late night / early morning) where automated monitoring is lower
- **Class imbalance confirmed:** 492 fraud cases in 284,807 
  transactions (0.17%) — drove all modelling and evaluation decisions

## 5. Machine Learning Models

### 5.1 XGBoost (Supervised)

- Handles class imbalance via `scale_pos_weight = |negatives| / |positives|`
- 300 estimators, `max_depth=6`, `learning_rate=0.05`
- Outputs a fraud probability in [0, 1]

### 5.2 Isolation Forest (Unsupervised)

- Label-free anomaly detection — learns the "normal" distribution
- `contamination=0.002` matches the known fraud rate
- Outputs −1 (anomaly) or +1 (normal), mapped to 1.0 / 0.0

---

## 6. Ensemble Strategy

```
Final Score = 0.70 × XGBoost_prob + 0.30 × IsoForest_score
```

The hybrid approach:
- Leverages labelled-data accuracy from XGBoost
- Adds generalization to novel, unseen fraud patterns via Isolation Forest
- Keeps the score bounded in [0, 1]

---

## 7. Risk Scoring

| Score Range | Risk Level | Action        |
|-------------|------------|---------------|
| > 0.90      | CRITICAL   | Alert + log   |
| 0.75 – 0.90 | HIGH       | Alert + log   |
| 0.50 – 0.75 | MEDIUM     | Log only      |
| ≤ 0.50      | LOW        | Log only      |

---

## 8. Alert System

Alerts fire for **CRITICAL** and **HIGH** risk levels:

```json
{
  "alert": true,
  "message": "Fraudulent transaction detected — risk level: CRITICAL, anomaly score: 0.9312"
}
```

---

## 9. Database Design

### Table: `transactions`

| Column     | Type                       | Notes              |
|------------|----------------------------|--------------------|
| id         | INTEGER (PK, auto)         | Unique record ID   |
| amount     | FLOAT                      | Raw transaction amount |
| score      | FLOAT                      | Ensemble anomaly score |
| risk       | VARCHAR(16)                | Risk tier (indexed)|
| created_at | TIMESTAMPTZ (server default)| UTC insert time   |

---

## 10. API Reference

### `GET /health`
Returns `{"status": "ok"}` — used by load-balancers and docker-compose healthchecks.

### `POST /detect`

**Request body** — all 30 original features:

```json
{
  "Time": 0.0,
  "V1": -1.36, "V2": -0.07, "V3": 2.54,
  "...": "...",
  "V28": -0.02,
  "Amount": 149.62
}
```

**Response:**

```json
{
  "transaction_id": 42,
  "score": 0.9312,
  "risk": "CRITICAL",
  "alert": {
    "alert": true,
    "message": "Fraudulent transaction detected — risk level: CRITICAL, anomaly score: 0.9312"
  }
}
```

Interactive docs available at `http://localhost:8000/docs` once the server is running.

---

## 11. Training Pipeline

```
creditcard.csv
      │
      ▼
StandardScaler on Amount  →  saved as models/scaler.pkl
      │
      ▼
Stratified 80/20 train/test split (random_state=42)
      │
      ├── XGBoost.fit(X_train, y_train)  →  models/xgb.pkl
      └── IsolationForest.fit(X_train)   →  models/iso.pkl
      │
      ▼
Evaluation: classification report + ROC-AUC (printed to stdout)
```

Run training:

```bash
python train.py
```

---

## 12. Model Performance

Evaluated on a stratified 20% hold-out set (56,962 transactions, 98 fraud cases).
Metrics chosen for class-imbalanced fraud detection (accuracy is misleading at a 0.17% base rate).

| Model                   | ROC-AUC | Precision | Recall | F1   |
|-------------------------|---------|-----------|--------|------|
| XGBoost                 | 0.9996  | 1.00      | 0.90   | 0.95 |
| Isolation Forest        | 0.6880  | 0.30      | 0.38   | 0.34 |
| **Ensemble (0.70/0.30)**| **0.9995** | **0.99** | **0.86** | **0.92** |

All metrics measured at a classification threshold of 0.50.

**Threshold analysis (risk tiers):**

- **CRITICAL (> 0.90):** maximises precision — every flagged transaction is genuine fraud; zero analyst false-alarm fatigue
- **HIGH (0.75–0.90):** catches borderline cases with an acceptable false-positive rate
- **At CRITICAL + HIGH thresholds: 100% precision on all analyst alerts (zero false alarms) covering 36.7% of fraud cases — the system prioritises analyst trust over raw coverage, ensuring every escalated case is genuine fraud

> The Isolation Forest contributes modest precision/recall in isolation (expected for an unsupervised model on a labelled benchmark) but improves ensemble generalisation to novel fraud patterns not seen during XGBoost training.

---

## 13. Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 15 (or use Docker)

### Local setup

```bash
# 1. Clone & enter the repo
git clone <repo-url>
cd Anomaly-Detection-System-main

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your DATABASE_URL

# 5. Train models  (requires creditcard.csv in project root)
python train.py

# 6. Start the API
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 14. Docker Deployment

```bash
# Build and start both services (API + PostgreSQL)
docker-compose up --build

# API available at http://localhost:8000
# PostgreSQL available at localhost:5432
```

The compose file includes a healthcheck on PostgreSQL so the API waits until the database is ready before accepting traffic.

---

## 15. Challenges & Solutions

| Challenge               | Solution                                  |
|-------------------------|-------------------------------------------|
| Extreme class imbalance | `scale_pos_weight` + stratified split     |
| Train/serve skew on Amount | Scaler persisted and reloaded at inference |
| Hard-coded credentials  | `DATABASE_URL` read from environment variable |
| Reproducible training   | `random_state=42` throughout              |
| Missing Dockerfile      | Added with multi-layer cache-friendly build |

---

## 16. Future Improvements

- **Autoencoder model** — reconstruction-error-based anomaly scoring
- **SHAP explainability** — per-prediction feature attribution
- **Kafka streaming** — replace HTTP polling with event-driven ingestion
- **Grafana dashboard** — live monitoring of score distributions and alert rates
- **Feedback-based retraining** — analyst labels feed back into XGBoost via online learning
- **A/B model serving** — shadow mode deployment for new model evaluation
