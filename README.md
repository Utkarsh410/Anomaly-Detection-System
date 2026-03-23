# Anomaly Detection System – Documentation Report

## 1. Project Overview

This project implements a **real-time anomaly detection system** for identifying fraudulent financial transactions using machine learning and statistical techniques. The system is designed to process incoming transaction data, assign a risk score, and generate alerts for suspicious activity.

The solution follows a **production-style fintech architecture**, combining:

* Machine Learning models (supervised + unsupervised)
* Risk scoring and alerting mechanisms
* Database storage for audit and investigation
* Real-time API for inference

---

## 2. Objectives

### Technical Objectives

* Implement anomaly detection using:

  * XGBoost (supervised)
  * Isolation Forest (unsupervised)
* Build a real-time inference API using FastAPI
* Store results in PostgreSQL

### Analytical Objectives

* Handle imbalanced fraud dataset
* Optimize precision and recall
* Perform feature scaling and preprocessing

### Operational Objectives

* Build alert generation system
* Enable investigation workflow via stored records
* Maintain system reliability and scalability

---

## 3. Dataset Description

**Dataset Used:** Credit Card Fraud Detection Dataset

| Feature | Description                          |
| ------- | ------------------------------------ |
| Time    | Seconds elapsed between transactions |
| V1–V28  | PCA-transformed features             |
| Amount  | Transaction amount                   |
| Class   | Target (0 = normal, 1 = fraud)       |

### Key Characteristics:

* Highly imbalanced dataset (0.17% fraud)
* 284,807 total transactions
* Suitable for anomaly detection benchmarking

---

## 4. System Architecture

### High-Level Flow

```
Transaction Input
        ↓
Feature Processing
        ↓
ML Models (XGBoost + Isolation Forest)
        ↓
Risk Scoring
        ↓
Alert Generation
        ↓
Database Storage
        ↓
API Response
```

---

## 5. Machine Learning Models

### 5.1 XGBoost (Supervised Model)

* Handles imbalanced data using `scale_pos_weight`
* Captures complex nonlinear relationships
* Outputs fraud probability score

### 5.2 Isolation Forest (Unsupervised Model)

* Detects anomalies without labels
* Identifies rare patterns via isolation mechanism
* Outputs anomaly flag (-1 or 1)

---

## 6. Ensemble Strategy

The final anomaly score is computed using a weighted combination:

```
Final Score = 0.7 × XGBoost + 0.3 × Isolation Forest
```

This hybrid approach improves:

* Detection accuracy
* Generalization to unseen fraud patterns

---

## 7. Risk Scoring System

Based on the final score:

| Score Range | Risk Level |
| ----------- | ---------- |
| > 0.90      | CRITICAL   |
| 0.75 – 0.90 | HIGH       |
| 0.50 – 0.75 | MEDIUM     |
| < 0.50      | LOW        |

This enables prioritization of alerts and efficient investigation.

---

## 8. Alert System

Alerts are generated for high-risk transactions:

* CRITICAL and HIGH → Alert triggered
* MEDIUM and LOW → Logged only

Example Alert:

```
Fraud detected with score 0.92
```

---

## 9. Database Design

### Table: Transactions

| Column | Type                  |
| ------ | --------------------- |
| id     | Integer (Primary Key) |
| amount | Float                 |
| score  | Float                 |
| risk   | String                |

Purpose:

* Store predictions
* Enable audit trails
* Support investigation workflows

---

## 10. API Design

### Endpoint: `/detect` (POST)

**Input:**

* Transaction features (V1–V28, Time, Amount)

**Output:**

```json
{
  "score": 0.91,
  "risk": "CRITICAL",
  "alert": {
    "alert": true,
    "message": "Fraud detected"
  }
}
```

---

## 11. Model Training Pipeline

Steps:

1. Load dataset
2. Scale features (Amount)
3. Split into train/test
4. Train XGBoost model
5. Train Isolation Forest
6. Save models using joblib

---

## 12. Evaluation Metrics

Due to class imbalance, the following metrics are used:

| Metric    | Purpose                |
| --------- | ---------------------- |
| Precision | Reduce false positives |
| Recall    | Detect fraud           |
| F1 Score  | Balance both           |
| ROC-AUC   | Overall performance    |

### Expected Performance:

* ROC-AUC: ~0.98–0.99
* Detection Rate: >85%
* False Positives: <5%

---

## 13. Deployment

### Local Deployment:

* Run FastAPI using Uvicorn
* PostgreSQL for storage

### Docker Deployment:

* API + Database using docker-compose
* Scalable and production-ready

---

## 14. Key Features

* Real-time fraud detection
* Ensemble ML approach
* Risk-based alert system
* Database-backed logging
* Modular architecture

---

## 15. Challenges & Solutions

| Challenge           | Solution              |
| ------------------- | --------------------- |
| Imbalanced data     | scale_pos_weight      |
| False positives     | risk threshold tuning |
| Feature scaling     | StandardScaler        |
| Real-time inference | FastAPI               |

---

## 16. Future Improvements

* Add Autoencoder model
* Implement Kafka streaming pipeline
* Introduce SHAP explainability
* Build monitoring dashboard (Grafana)
* Implement feedback-based retraining

---

## 17. Conclusion

This project successfully demonstrates a **production-style anomaly detection system** combining:

* Machine learning
* Statistical techniques
* Real-time APIs
* Alerting workflows

It aligns with industry practices used in:

* Banking fraud detection
* Payment systems
* Trading surveillance

The system is scalable, extensible, and suitable for real-world deployment.

---
