# FinSentinel — AI-Powered Financial Fraud Detection

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B)
![AUROC](https://img.shields.io/badge/AUROC-98.1%25-brightgreen)

An end-to-end fraud detection system combining supervised and unsupervised machine learning with real-time inference, explainability, and a live monitoring dashboard.

---

## Model Performance

| Model | AUROC | Fraud Recall | Fraud Precision |
|-------|-------|-------------|-----------------|
| XGBoost + Isolation Forest | **0.9810** | 87% | 82% |
| LSTM Autoencoder | 0.9554 | 82% | 58% |

Trained on the [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 transactions with a 0.17% fraud rate (highly imbalanced).

---

## Architecture

```
Raw Data (creditcard.csv)
        │
        ▼
  preprocess.py        Feature engineering, standard scaling, stratified split
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
   model.py                          lstm_autoencoder.py
   XGBoost Classifier                LSTM Autoencoder
   + Isolation Forest                Unsupervised anomaly detection
   (supervised)                      (reconstruction error threshold)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
                  explain.py
                  SHAP TreeExplainer
                  Feature importance & per-prediction explanations
                       │
                       ▼
                   api.py
                   FastAPI REST endpoint
                   /predict → fraud probability + risk level + top reasons
                       │
             ┌─────────┴──────────┐
             ▼                    ▼
      dashboard.py          Kafka Streaming
      Streamlit UI          Producer → Consumer → API
      Real-time testing     (Dockerized)
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| ML Models | XGBoost, Scikit-learn |
| Deep Learning | TensorFlow / Keras (LSTM Autoencoder) |
| Anomaly Detection | Isolation Forest, LSTM Reconstruction Error |
| Explainability | SHAP (TreeExplainer) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Streaming | Apache Kafka (kafka-python) |
| Deployment | Docker, Docker Compose |

---

## Project Structure

```
FinSentinel/
├── src/
│   ├── preprocess.py          # Data pipeline: feature engineering, scaling, split
│   ├── model.py               # XGBoost + Isolation Forest training
│   ├── lstm_autoencoder.py    # LSTM Autoencoder training and threshold detection
│   ├── explain.py             # SHAP summary and beeswarm plots
│   ├── api.py                 # FastAPI inference endpoint
│   ├── kafka_producer.py      # Simulates real-time transaction stream
│   └── kafka_consumer.py      # Consumes stream, calls prediction API
├── app/
│   └── dashboard.py           # Streamlit monitoring dashboard
├── models/                    # Serialized models (not tracked)
├── outputs/                   # Generated plots (not tracked)
├── data/                      # Raw dataset (not tracked)
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Setup & Usage

### 1. Clone and install dependencies

```bash
git clone https://github.com/TishaRathore11/FinSentinel.git
cd FinSentinel
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory.

### 3. Train models

```bash
cd src
python preprocess.py
python model.py
python lstm_autoencoder.py
python explain.py
```

### 4. Start the API and dashboard

```bash
# Terminal 1
uvicorn api:app --reload --port 8000

# Terminal 2
streamlit run app/dashboard.py
```

- Dashboard: http://localhost:8501  
- API documentation: http://localhost:8000/docs

---

## Key Design Decisions

**Hybrid detection approach** — XGBoost handles labeled fraud patterns while the LSTM Autoencoder detects distributional anomalies without requiring labels. The Isolation Forest anomaly score is injected as an engineered feature into XGBoost.

**Class imbalance handling** — `scale_pos_weight` in XGBoost is set to the ratio of normal to fraud samples (~577:1), preventing the model from defaulting to predicting all transactions as normal.

**Explainability** — SHAP TreeExplainer generates both global feature importance and per-transaction explanations, making predictions auditable and production-ready.

**Threshold optimization** — LSTM threshold is set at the 99.9th percentile of normal transaction reconstruction error on the test set, minimizing false positives.

---

## API Reference

**POST** `/predict`

Request body: transaction features (V1–V28, Amount_log, Hour)

Response:
```json
{
  "fraud_probability": 0.9231,
  "is_fraud": true,
  "risk_level": "HIGH",
  "top_reasons": [
    {"feature": "V14", "impact": -0.8421},
    {"feature": "V4",  "impact":  0.6103},
    {"feature": "V12", "impact": -0.4872}
  ]
}
```

---

## Author

**Tisha Rathore** — B.Tech, Electronics & Communication Technology  
[GitHub](https://github.com/TishaRathore11)
