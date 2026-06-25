import streamlit as st
import requests, json
import pandas as pd
import plotly.express as px
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="FinSentinel",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ FinSentinel — Fraud Detection Dashboard")
st.markdown("Real-time credit card fraud detection with explainability")

# ─── Sidebar: Manual Test ─────────────────────────────
st.sidebar.header("🔍 Test a Transaction")
amount = st.sidebar.number_input("Amount ($)", 1.0, 25000.0, 150.0)
hour   = st.sidebar.slider("Hour of Day", 0, 23, 14)

if st.sidebar.button("Check Fraud", type="primary"):

    txn = {
        'Amount_log': float(np.log1p(amount)),
        'Hour': float(hour),
        'anomaly_score': 0.0
    }

    for i in range(1, 29):
        txn[f'V{i}'] = float(np.random.normal(0, 1))

    with st.spinner("Analyzing..."):

        try:
            res = requests.post(
                "http://localhost:8000/predict",
                json=txn
            )

            r = res.json()

            # ─── Metrics ─────────────────────
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Fraud Probability",
                f"{r['fraud_probability']*100:.2f}%"
            )

            col2.metric(
                "Status",
                "🔴 FRAUD" if r['is_fraud']
                else "🟢 LEGIT"
            )

            col3.metric(
                "Risk Level",
                r['risk_level']
            )

            # ─── Reasons chart ──────────────
            st.subheader("Why flagged?")

            reasons_df = pd.DataFrame(
                r['top_reasons']
            )

            fig = px.bar(
                reasons_df,
                x='feature',
                y='impact',
                color='impact',
                color_continuous_scale='RdYlGn_r',
                title='Top 3 Contributing Features'
            )

            st.plotly_chart(
                fig,
                width='stretch'
            )

        except Exception as e:
            st.error(
                f" API Error: {e}"
            )

# ─── SHAP plots dikhao ────────────────────────────────
st.subheader("📊 Model Insights")
col1, col2 = st.columns(2)
try:
    col1.image('outputs/shap_bar.png', 
               caption='SHAP Feature Importance')
    col2.image('outputs/confusion_matrix.png',
               caption='Confusion Matrix')
except:
    st.info("Run model training first to see plots")