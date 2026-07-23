import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Inferencing Function
# ---------------------------
def predict_house_price(input_features):

    # Load Model
    with open(r'C:\Users\inter\Desktop\Shrutika\Python-Practical\ML regression project house price prediction\best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # Load Scaler
    with open(r'C:\Users\inter\Desktop\Shrutika\Python-Practical\ML regression project house price prediction\scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Convert input into NumPy array
    input_array = np.array(input_features).reshape(1, -1)

    # Scale input data
    input_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_scaled)

    return prediction[0]


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🏠 House Price Prediction App")

st.write("Enter House Details")

# Replace these fields with your actual dataset columns
feature1 = st.number_input("Room Number", min_value=1, max_value=10, value=10)
feature2 = st.number_input("Overall Quality", min_value=1, max_value=10, value=5)
feature3 = st.number_input("Ground Living Area", value=1500)
feature4 = st.number_input("Garage Cars", value=2)
feature5 = st.number_input("Total Basement Area", value=800)
feature6 = st.number_input("Full Bathrooms", value=2)
feature7 = st.number_input("Bedrooms Above Ground", value=3)
feature8 = st.number_input("Year Built", value=2000)
feature9 = st.number_input("Remodel area",value=2025)
feature10 = st.number_input("Lot Area", value=8000)
feature11 = st.number_input("Guesthouseroom",value=8000)
feature12 = st.number_input("Balcony",value=2000)

if st.button("Predict House Price"):

    input_features = [
        feature1,
        feature2,
        feature3,
        feature4,
        feature5,
        feature6,
        feature7,
        feature8,
        feature9,
        feature10,
        feature11,
        feature12
    ]

    result = predict_house_price(input_features)

    st.success(f"Predicted House Price: {result:,.2f}")