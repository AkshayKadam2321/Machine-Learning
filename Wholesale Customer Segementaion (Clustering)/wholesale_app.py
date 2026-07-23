import streamlit as st
import pickle
import pandas as pd

# -----------------------------
# Load Model
# -----------------------------
with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML clustering project wholesale customer\kmeans_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load Scaler (if used)
with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML clustering project wholesale customer\scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# -----------------------------
# Title
# -----------------------------
st.set_page_config(page_title="Wholesale Customer Segmentation")

st.title("🛒 Wholesale Customer Segmentation")
st.write("Enter the customer's annual spending to predict the cluster.")

# -----------------------------
# User Inputs
# -----------------------------
fresh = st.number_input("Fresh", min_value=0.0)
milk = st.number_input("Milk", min_value=0.0)
grocery = st.number_input("Grocery", min_value=0.0)
frozen = st.number_input("Frozen", min_value=0.0)
detergents = st.number_input("Detergents_Paper", min_value=0.0)
delicassen = st.number_input("Delicassen", min_value=0.0)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Cluster"):

    input_data = pd.DataFrame({
        "Fresh": [fresh],
        "Milk": [milk],
        "Grocery": [grocery],
        "Frozen": [frozen],
        "Detergents_Paper": [detergents],
        "Delicassen": [delicassen]
    })

    # Scale Data
    input_scaled = scaler.transform(input_data)

    # Predict
    cluster = model.predict(input_scaled)[0]

    st.success(f"Predicted Cluster : {cluster}")

    if cluster == 0:
        st.info("Regular Customer")
    elif cluster == 1:
        st.info("Retail Customer")
    else:
        st.info("High-Value Customer")