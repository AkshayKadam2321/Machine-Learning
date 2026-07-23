import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Load Model and Scaler
# -------------------------------
@st.cache_resource
def load_model():
    with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML classsification project Loan Prediction\loan_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)

    with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML classsification project Loan Prediction\loan_scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


model, scaler = load_model()

# -------------------------------
# App Title
# -------------------------------
st.set_page_config(page_title="Loan Approval Prediction", page_icon="💰")

st.title("💰 Loan Approval Prediction")
st.write("Enter the applicant details below.")

# -------------------------------
# User Inputs
# -------------------------------
loan_id = st.number_input("Loan ID", min_value=1, value=1)

no_of_dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=10,
    value=0
)

education = st.selectbox(
    "Education",
    ["Graduate", "Not Graduate"]
)

self_employed = st.selectbox(
    "Self Employed",
    ["Yes", "No"]
)

income_annum = st.number_input(
    "Annual Income",
    min_value=0,
    value=5000000
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000000
)

loan_term = st.number_input(
    "Loan Term (Years)",
    min_value=1,
    value=15
)

cibil_score = st.number_input(
    "CIBIL Score",
    min_value=300,
    max_value=900,
    value=750
)

residential_assets_value = st.number_input(
    "Residential Assets Value",
    min_value=0,
    value=6000000
)

commercial_assets_value = st.number_input(
    "Commercial Assets Value",
    min_value=0,
    value=2000000
)

luxury_assets_value = st.number_input(
    "Luxury Assets Value",
    min_value=0,
    value=3000000
)

bank_asset_value = st.number_input(
    "Bank Asset Value",
    min_value=0,
    value=1500000
)

# -------------------------------
# Encode Categorical Features
# -------------------------------
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Loan Status"):

    input_df = pd.DataFrame([{
        "loan_id": loan_id,
        "no_of_dependents": no_of_dependents,
        "education": education,
        "self_employed": self_employed,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value
    }])

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")