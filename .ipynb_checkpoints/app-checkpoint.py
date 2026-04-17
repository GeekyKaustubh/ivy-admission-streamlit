import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("best_model.pkl")

st.title("🎯 Model Prediction App")

# Input fields using your actual training column names
gre_score = st.number_input("GRE Score (Limit- 260 ~ 340)", value=0.0)
toefl_score = st.number_input("TOEFL Score (Limit- 0 ~ 120)", value=0.0)
cgpa = st.number_input("CGPA (Limit- 0.0 ~ 10.0)", value=0.0)

# Convert input to DataFrame with exact same column names as training
input_df = pd.DataFrame([[gre_score, toefl_score, cgpa]],
                        columns=["GRE Score", "TOEFL Score", "CGPA"])

# Predict
if st.button("Chances of Admission Prediction"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Value: {prediction[0]:.4f}")
