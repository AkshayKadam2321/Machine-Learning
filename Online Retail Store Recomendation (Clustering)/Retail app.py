import streamlit as st
import pickle
import numpy as np

# --------------------------------------
# Load Model and Scaler
# --------------------------------------
with open(r"C:\Users\inter\Desktop\Shrutika\ML Project\Online Retail Store Recomendation (Clustering)\kmeans_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\inter\Desktop\Shrutika\ML Project\Online Retail Store Recomendation (Clustering)\scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --------------------------------------
# Streamlit Page Configuration
# --------------------------------------
st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Customer Segmentation using K-Means")
st.markdown("Enter the customer's **RFM values** to predict the customer segment.")

st.divider()

# --------------------------------------
# Input Fields
# --------------------------------------
recency = st.number_input(
    "Recency (Days)",
    min_value=0,
    value=20
)

frequency = st.number_input(
    "Frequency",
    min_value=1,
    value=10
)

monetary = st.number_input(
    "Monetary Value",
    min_value=0.0,
    value=5000.0
)

# --------------------------------------
# Prediction
# --------------------------------------
if st.button("Predict Customer Segment"):

    customer = np.array([[recency, frequency, monetary]])

    customer_scaled = scaler.transform(customer)

    cluster = model.predict(customer_scaled)[0]

    st.success(f"Predicted Cluster : {cluster}")

    # -------------------------
    # Cluster Interpretation
    # -------------------------
    n_clusters = model.n_clusters

    if n_clusters == 2:

        if cluster == 0:
            st.success("🟢 High Value Customer")

        else:
            st.warning("🟡 Low Value Customer")

    elif n_clusters == 4:

        segments = {
            0: "👑 VIP Customer",
            1: "🙂 Regular Customer",
            2: "🆕 New Customer",
            3: "💰 Low Value Customer"
        }

        st.info(f"Customer Segment : {segments.get(cluster)}")

    else:

        st.info(f"Customer belongs to Cluster {cluster}")

# --------------------------------------
# Footer
# --------------------------------------
st.divider()
st.caption("Customer Segmentation Project using K-Means Clustering")