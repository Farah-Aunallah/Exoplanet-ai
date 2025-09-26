import streamlit as st
import pandas as pd
import joblib

# Load saved model + preprocessing
model = joblib.load("exoplanet_model.pkl")
imputer = joblib.load("imputer.pkl")
features = joblib.load("features.pkl")

st.title("🪐 Exoplanet Confirmation AI")

st.write("Enter candidate data, and the AI predicts if it's a confirmed exoplanet.")

# Input fields for each feature
user_input = {}
for feat in features:
    user_input[feat] = st.number_input(f"{feat}", value=0.0)

# Turn into DataFrame
input_df = pd.DataFrame([user_input])

# Apply imputer
input_df = pd.DataFrame(imputer.transform(input_df), columns=features)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("✅ This is likely a CONFIRMED exoplanet!")
    else:
        st.error("❌ This is likely a FALSE candidate.")
