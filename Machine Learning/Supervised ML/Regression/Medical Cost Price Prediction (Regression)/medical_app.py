import streamlit as st
import numpy as np
import pickle

# -------------------------------
# Load Model and Scaler
# -------------------------------

with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML regression project medical cost prediction\best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML regression project medical cost prediction\scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)


# -------------------------------
# Prediction Function
# -------------------------------

def predict_insurance(age, sex, bmi, children, smoker, region):

    input_data = np.array([[age,
                            sex,
                            bmi,
                            children,
                            smoker,
                            region]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    return prediction[0]


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Medical Insurance Cost Prediction", layout="centered")

st.title("🏥 Medical Insurance Cost Prediction")

st.write("Enter the details below to predict the insurance charges.")

st.sidebar.header("Input Features")

# Age
age = st.sidebar.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=25
)

# Sex
sex_option = st.sidebar.selectbox(
    "Sex",
    ("Female", "Male")
)

sex = 0 if sex_option == "Female" else 1

# BMI
bmi = st.sidebar.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

# Children
children = st.sidebar.number_input(
    "Children",
    min_value=0,
    max_value=10,
    value=0
)

# Smoker
smoker_option = st.sidebar.selectbox(
    "Smoker",
    ("No", "Yes")
)

smoker = 0 if smoker_option == "No" else 1

# Region
region_option = st.sidebar.selectbox(
    "Region",
    (
        "Northeast",
        "Northwest",
        "Southeast",
        "Southwest"
    )
)

region_dict = {
    "Northeast": 0,
    "Northwest": 1,
    "Southeast": 2,
    "Southwest": 3
}

region = region_dict[region_option]


# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Insurance Cost"):

    result = predict_insurance(
        age,
        sex,
        bmi,
        children,
        smoker,
        region
    )

    st.success(f"Predicted Insurance Charge: ${result:,.2f}")