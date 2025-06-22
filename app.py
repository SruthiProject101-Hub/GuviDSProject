import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and PCA
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

st.title("Voice Gender Prediction App")

uploaded_file = st.file_uploader("Upload CSV with voice features", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", data.head())

    if st.button("Predict"):
        try:
            # Apply preprocessing
            data_scaled = scaler.transform(data)
            data_pca = pca.transform(data_scaled)

            # Predict
            predictions = model.predict(data_pca)
            prediction_labels = ["Female" if p == 0 else "Male" for p in predictions]

            st.write("Predictions:")
            st.write(prediction_labels)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
