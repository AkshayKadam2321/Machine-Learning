# app.py

import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("stock_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("📈 Stock Price Prediction App")

st.write("Predict whether stock price will go UP or DOWN")

open_price = st.number_input("Open Price", value=100.0)
close_price = st.number_input("Close Price", value=100.0)
low_price = st.number_input("Low Price", value=95.0)
high_price = st.number_input("High Price", value=105.0)

quarter = st.selectbox(
    "Is Quarter End?",
    [0, 1]
)

if st.button("Predict"):

    open_close = open_price - close_price
    low_high = low_price - high_price

    data = np.array([[open_close,
                      low_high,
                      quarter]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    if prediction == 1:
        st.success("📈 Stock Price Will Go UP")
    else:
        st.error("📉 Stock Price Will Go DOWN")