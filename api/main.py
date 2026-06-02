from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn prediction using XGBoost",
    version="1.0.0"
)

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model = joblib.load(os.path.join(BASE_DIR, "outputs", "model.pkl"))

class CustomerInput(BaseModel):
    tenure: float
    MonthlyCharges: float
    TotalCharges: float
    SeniorCitizen: int
    Partner: int
    Dependents: int
    PhoneService: int
    MultipleLines: int
    OnlineSecurity: int
    OnlineBackup: int
    DeviceProtection: int
    TechSupport: int
    StreamingTV: int
    StreamingMovies: int
    PaperlessBilling: int
    gender: int
    Contract_Month_to_month: int
    Contract_One_year: int
    Contract_Two_year: int
    InternetService_DSL: int
    InternetService_Fiber_optic: int
    InternetService_No: int
    PaymentMethod_Bank_transfer: int
    PaymentMethod_Credit_card: int
    PaymentMethod_Electronic_check: int
    PaymentMethod_Mailed_check: int

def engineer_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Tenure group
    def tenure_group(t):
        if t <= 12:   return 0
        elif t <= 24: return 1
        elif t <= 48: return 2
        else:         return 3

    df['tenure_group'] = df['tenure'].apply(tenure_group)
    df['avg_monthly_spend_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
    df['support_score'] = (df['OnlineSecurity'] + df['TechSupport'] +
                           df['OnlineBackup'] + df['DeviceProtection'])
    df['streaming_count'] = df['StreamingTV'] + df['StreamingMovies']
    df['is_high_value'] = (df['MonthlyCharges'] > 65).astype(int)
    df['engagement_score'] = (df['PhoneService'] + df['MultipleLines'] +
                               df['streaming_count'] + df['support_score'])

    # Rename columns to match training
    df.rename(columns={
        'Contract_Month_to_month': 'Contract_Month-to-month',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year',
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'PaymentMethod_Bank_transfer': 'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit_card': 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check'
    }, inplace=True)

    # Align with training columns
    training_cols = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "features.csv")).drop('Churn', axis=1).columns
    df = df.reindex(columns=training_cols, fill_value=0)
    return df

@app.get("/")
def root():
    return {
        "status": "running",
        "api": "Customer Churn Prediction API",
        "version": "1.0.0"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost",
        "auc_roc": 0.8461
    }

@app.post("/predict")
def predict(customer: CustomerInput):
    data = customer.dict()
    df = engineer_features(data)

    prob = float(model.predict_proba(df)[0][1])
    predicted = int(prob >= 0.5)

    if prob < 0.3:
        risk_tier = "Low Risk"
        action = "No immediate action required. Standard engagement."
    elif prob < 0.6:
        risk_tier = "Medium Risk"
        action = "Proactive outreach recommended. Check in with customer."
    else:
        risk_tier = "High Risk"
        action = "Immediate intervention required. Offer loyalty discount."

    return {
        "churn_probability": round(prob, 4),
        "churn_probability_pct": f"{prob*100:.1f}%",
        "predicted_churn": bool(predicted),
        "risk_tier": risk_tier,
        "recommended_action": action
    }