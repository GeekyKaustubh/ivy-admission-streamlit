import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("best_model.pkl")

st.title("🎯 Ivy League Adimission Prediction 📚")

# Input fields with strict validation
gre_score = st.number_input("GRE Score (260–340)", min_value=260, max_value=340, value=300)
toefl_score = st.number_input("TOEFL Score (0–120)", min_value=0, max_value=120, value=100)
university_rating = st.number_input("University Rating (1–5)", min_value=1, max_value=5, value=3)
sop = st.number_input("SOP (1–5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
lor = st.number_input("LOR (1–5)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
cgpa = st.number_input("CGPA (0.0–10.0)", min_value=0.0, max_value=10.0, value=8.0, step=0.1)
research = st.number_input("Research (0 or 1)", min_value=0, max_value=1, value=0)

# Create input in same order as training
input_df = pd.DataFrame([[
    gre_score, toefl_score, university_rating,
    sop, lor, cgpa, research
]], columns=model.feature_names_in_)

# Predict
if st.button("Chances of Admission Prediction"):
    prediction = model.predict(input_df)
    st.success(f"Predicted Value: {prediction[0]:.4f}")
