from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import shap

app = FastAPI(title="FinSentinel", description="Real-Time Fraud Detection API", version="1.0.0")

model      = joblib.load('models/xgboost_model.pkl')
iso_forest = joblib.load('models/isolation_forest.pkl')
scaler     = joblib.load('models/scaler.pkl')
explainer  = shap.TreeExplainer(model)


class Transaction(BaseModel):
    V1: float;  V2: float;  V3: float;  V4: float
    V5: float;  V6: float;  V7: float;  V8: float
    V9: float;  V10: float; V11: float; V12: float
    V13: float; V14: float; V15: float; V16: float
    V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float
    V25: float; V26: float; V27: float; V28: float
    Amount_log: float
    Hour: float
    anomaly_score: float = 0.0


@app.get("/")
def root():
    return {"status": "FinSentinel API running"}


@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost v1"}


@app.post("/predict")
def predict(txn: Transaction):
    try:
        data = pd.DataFrame([txn.dict()])
        data['anomaly_score'] = iso_forest.decision_function(data.drop('anomaly_score', axis=1))

        prob     = float(model.predict_proba(data)[0][1])
        is_fraud = prob > 0.5

        shap_values = explainer.shap_values(data)
        top_reasons = sorted(
            zip(data.columns, shap_values[0]),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        return {
            "fraud_probability": round(prob, 4),
            "is_fraud": is_fraud,
            "risk_level": "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.4 else "LOW",
            "top_reasons": [{"feature": f, "impact": round(float(v), 4)} for f, v in top_reasons]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))