import streamlit as st
import pandas as pd
import numpy as np
import pickle
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
h1 {
    color: red;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Page config
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Title
st.markdown("<h1 style='text-align: center; color: red;'>💓 Heart Disease 10-Year Risk Predictor</h1>", unsafe_allow_html=True)

st.write("### Enter Patient Clinical Details")

# --- Input Section ---
col1, col2 = st.columns(2)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 20, 100, 30)
    
    gender_label = st.selectbox("Gender", ["Female", "Male"])
    gender = 1 if gender_label == "Female" else 2  # mapping

    height = st.slider("Height (cm)", 100, 220, 170)
    weight = st.slider("Weight (kg)", 30, 150, 70)

    ap_hi = st.slider("Systolic BP (ap_hi)", 80, 200, 120)
    ap_lo = st.slider("Diastolic BP (ap_lo)", 50, 150, 80)

with col2:
    cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
    glucose = st.selectbox("Glucose Level", [1, 2, 3])

    smoke = st.selectbox("Smoking", [0, 1])
    alco = st.selectbox("Alcohol Intake", [0, 1])
    active = st.selectbox("Physical Activity", [0, 1])

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Risk"):

    input_df = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "height": height,
    "weight": weight,
    "ap_hi": ap_hi,
    "ap_lo": ap_lo,
    "cholesterol": cholesterol,
    "gluc": glucose,   
    "smoke": smoke,
    "alco": alco,
    "active": active
}])

    input_df = input_df[[
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol",
    "gluc", "smoke", "alco", "active"
]]

    # Scale input
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    # Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # -------------------------------
    # Output
    # -------------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK of Heart Disease\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ LOW RISK of Heart Disease\nProbability: {probability:.2f}")

    # -------------------------------
    # Health Insights
    # -------------------------------
    st.subheader("Health Insights")

    if ap_hi > 140:
        st.warning("⚠️ High Systolic Blood Pressure detected!")

    if ap_lo > 90:
        st.warning("⚠️ High Diastolic Blood Pressure detected!")

    if cholesterol == 3:
        st.warning("⚠️ High Cholesterol level!")

    if gluc == 3:
        st.warning("⚠️ High Glucose level!")

    if active == 0:
        st.warning("⚠️ Low Physical Activity detected!")
    
   
