import streamlit as st
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

with col1:
    age = st.slider("Age", 20, 100, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    smoke = st.selectbox("Smoking", ["No", "Yes"])
    bmi = st.slider("BMI", 10.0, 50.0, 22.0)

with col2:
    bp = st.slider("Systolic BP", 80, 200, 120)
    chol = st.slider("Cholesterol", 100, 400, 200)
    glucose = st.slider("Glucose", 70, 300, 100)
    heart_rate = st.slider("Heart Rate", 40, 180, 75)

# Convert categorical
sex = 1 if sex == "Male" else 0
smoke = 1 if smoke == "Yes" else 0

# --- Prediction ---
if st.button("🔍 Predict Risk"):

    input_data = np.array([[age, sex, bp, chol, glucose, smoke, bmi, heart_rate]])
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ HIGH RISK of Heart Disease in 10 Years\nProbability: {probability:.2f}")
    else:
        st.success(f"✅ LOW RISK of Heart Disease\nProbability: {probability:.2f}")

    # --- Smart Insights ---
    st.subheader("Health Insights")

    if bp > 140:
        st.warning("High Blood Pressure detected!")

    if chol > 240:
        st.warning("High Cholesterol detected!")

    if bmi > 30:
        st.warning("Obesity risk detected!")

