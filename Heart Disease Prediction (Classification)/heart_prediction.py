import streamlit as st
import pickle
import pandas as pd

# --------------------------------------------
# Load Model and Scaler
# --------------------------------------------
def load_model_and_scaler(
    model_filename='heart_model.pkl',
    scaler_filename='heart_scaler.pkl'
):

    with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML Project 3 Heart Disease Prediction\heart_model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

    with open(r"C:\Users\inter\Desktop\Shrutika\Python-Practical\ML Project 3 Heart Disease Prediction\heart_scaler.pkl", 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


# --------------------------------------------
# Preprocess Input Data
# --------------------------------------------
def preprocess_input(input_data, scaler):

    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)

    return input_scaled


# --------------------------------------------
# Predict Heart Disease
# --------------------------------------------
def predict_heart_disease(input_data):

    model, scaler = load_model_and_scaler()

    input_scaled = preprocess_input(input_data, scaler)

    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        return "Heart Disease Detected"
    else:
        return "No Heart Disease"


# --------------------------------------------
# Streamlit App Interface
# --------------------------------------------
def main():

    st.title("❤️ Heart Disease Prediction App")

    st.write("Enter patient details below:")

    age = st.number_input("Age", min_value=1, max_value=120, value=50)

    sex = st.selectbox(
        "Sex",
        [0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male"
    )

    cp = st.number_input("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)

    trestbps = st.number_input(
        "Resting Blood Pressure (trestbps)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Cholesterol (chol)",
        min_value=50,
        max_value=600,
        value=200
    )

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])

    restecg = st.number_input(
        "Resting ECG Results (restecg)",
        min_value=0,
        max_value=2,
        value=0
    )

    thalach = st.number_input(
        "Maximum Heart Rate (thalach)",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0
    )

    slope = st.number_input(
        "Slope",
        min_value=0,
        max_value=2,
        value=1
    )

    ca = st.number_input(
        "Number of Major Vessels (ca)",
        min_value=0,
        max_value=4,
        value=0
    )

    thal = st.number_input(
        "Thal",
        min_value=0,
        max_value=3,
        value=2
    )

    # Prediction Button
    if st.button("Predict"):

        input_data = {
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }

        result = predict_heart_disease(input_data)

        st.success(f"Prediction Result: {result}")


# --------------------------------------------
# Run App
# --------------------------------------------
if __name__ == "__main__":
    main()