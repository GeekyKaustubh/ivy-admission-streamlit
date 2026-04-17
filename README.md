# Ivy League Admission Predictor

A Streamlit web app that predicts the chance of admission based on applicant profile features such as GRE, TOEFL, SOP, LOR, CGPA, university rating, and research experience.

## Features
- Interactive Streamlit UI
- Loads trained ML model from `best_model.pkl`
- Predicts admission probability
- Simple and portfolio-friendly deployment

## Project Structure
.
├── streamlit_app.py
├── best_model.pkl
├── requirements.txt
└── README.md

## Run Locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py