# streamlit_app.py
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Claim Amount Predictor", page_icon="💷", layout="centered")

st.title("💷 Ultimate Claim Amount – Predictor")
st.caption("Enter features for a single day/claim or upload a CSV to get predicted values.")

@st.cache_resource
def load_pipeline(path: str):
    return joblib.load(path)

def compute_reporting_lag(df: pd.DataFrame) -> pd.DataFrame:
    if "Reporting_Lag" not in df.columns:
        # Try to derive from Accident_Date and FNOL_Date if present
        if {"Accident_Date", "FNOL_Date"}.issubset(df.columns):
            ad = pd.to_datetime(df["Accident_Date"], dayfirst=True, errors="coerce")
            fn = pd.to_datetime(df["FNOL_Date"], dayfirst=True, errors="coerce")
            df["Reporting_Lag"] = (fn - ad).dt.days.clip(lower=0)
    return df

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Required columns expected by the model
    num_features = ["Age_of_Driver", "Annual_Mileage", "Driving_Experience_Years", "Vehicle_Age", "Reporting_Lag"]
    cat_features = [
        "Claim_Type", "Claim_Complexity", "Fraud_Flag", "Litigation_Flag", "Severity_Band",
        "Gender", "Occupation", "Region", "Vehicle_Type", "Credit_Score_Band"
    ]

    # Create any missing columns with NA
    for c in num_features + cat_features:
        if c not in df.columns:
            df[c] = np.nan

    # Reorder
    return df[num_features + cat_features]

def predict_amount(pipeline, X: pd.DataFrame) -> np.ndarray:
    # Model was trained on log1p(target); we back-transform here
    log_preds = pipeline.predict(X)
    return np.expm1(log_preds)

# Sidebar – model path
model_path = st.sidebar.text_input("Path to saved model (.joblib)", value="best_claim_model.joblib")
try:
    clf = load_pipeline(model_path)
    st.sidebar.success("Model loaded.")
except Exception as e:
    st.sidebar.error(f"Could not load model: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Single prediction", "Batch prediction (CSV)"])

with tab1:
    st.subheader("Single prediction")
    st.write("Enter features (or provide dates to auto-calculate **Reporting_Lag**).")

    col1, col2 = st.columns(2)

    with col1:
        accident_date = st.date_input("Accident_Date", value=None)
        fnol_date = st.date_input("FNOL_Date", value=None)
        age = st.number_input("Age_of_Driver", min_value=16, max_value=100, value=35)
        mileage = st.number_input("Annual_Mileage", min_value=0, max_value=200_000, value=12000, step=500)
        exp_years = st.number_input("Driving_Experience_Years", min_value=0, max_value=80, value=10)
        vehicle_age = st.number_input("Vehicle_Age", min_value=0, max_value=30, value=5)

    with col2:
        claim_type = st.selectbox("Claim_Type", ["Collision", "Fire", "Other", "Theft", "Vandalism", "Weather"])
        claim_complexity = st.selectbox("Claim_Complexity", ["Low", "Medium", "High"])
        fraud_flag = st.selectbox("Fraud_Flag", ["No", "Yes"])
        litigation_flag = st.selectbox("Litigation_Flag", ["No", "Yes"])
        severity_band = st.selectbox("Severity_Band", ["Low", "Medium", "High", "Catastrophic"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
        occupation = st.text_input("Occupation", value="")
        region = st.text_input("Region", value="")
        vehicle_type = st.selectbox("Vehicle_Type", ["Saloon", "Hatchback", "SUV", "Truck", "Van", "Other"])
        credit_band = st.selectbox("Credit_Score_Band", ["Poor", "Fair", "Good", "Very Good", "Excellent"])

    # Calculate reporting lag
    reporting_lag = None
    if accident_date and fnol_date:
        try:
            reporting_lag = max((pd.to_datetime(fnol_date) - pd.to_datetime(accident_date)).days, 0)
        except Exception:
            reporting_lag = None

    # Build input row
    row = {
        "Age_of_Driver": age,
        "Annual_Mileage": mileage,
        "Driving_Experience_Years": exp_years,
        "Vehicle_Age": vehicle_age,
        "Reporting_Lag": reporting_lag if reporting_lag is not None else st.number_input("Reporting_Lag (days)", min_value=0, value=2),
        "Claim_Type": claim_type,
        "Claim_Complexity": claim_complexity,
        "Fraud_Flag": fraud_flag,
        "Litigation_Flag": litigation_flag,
        "Severity_Band": severity_band,
        "Gender": gender,
        "Occupation": occupation,
        "Region": region,
        "Vehicle_Type": vehicle_type,
        "Credit_Score_Band": credit_band,
    }
    X = pd.DataFrame([row])

    if st.button("Predict"):
        try:
            pred = float(predict_amount(clf, X)[0])
            st.success(f"Predicted Ultimate_Claim_Amount: £{pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch prediction (CSV)")
    st.write("Upload a CSV. **Do Not Include Policy and Claim ID")

    file = st.file_uploader("CSV with columns for features", type=["csv"])
    if file is not None:
        try:
            raw_df = pd.read_csv(file)
            df = compute_reporting_lag(raw_df.copy())
            X = ensure_columns(df)
            preds = predict_amount(clf, X)
            out = df.copy()
            out["Predicted_Ultimate_Claim_Amount"] = preds

            st.write("Preview:")
            st.dataframe(out.head(20))

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Failed to score file: {e}")
