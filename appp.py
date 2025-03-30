import streamlit as st
import numpy as np
import joblib

# Load models and scalers
diseases = ["Parkinson’s", "Diabetes", "Breast Cancer", "Heart Disease", "Hypertension"]
models = {}
scalers = {}

for disease in diseases:
    model_filename = f"model_{disease.lower().replace(' ', '_')}.joblib"
    scaler_filename = f"scaler_{disease.lower().replace(' ', '_')}.joblib"
   
    models[disease] = joblib.load(model_filename)
    scalers[disease] = joblib.load(scaler_filename)

st.title("Health Prediction App")
st.write("Select a disease type and enter the required details.")

selected_disease = st.selectbox("Select Disease Type:", diseases)

# Define input fields based on selected disease
input_fields = {
    "Diabetes": ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"],
    "Parkinson’s": ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "Shimmer", "HNR", "RPDE", "DFA"],
    "Breast Cancer": ["Radius Mean", "Texture Mean", "Perimeter Mean", "Area Mean", "Smoothness Mean", "Compactness Mean", "Concavity Mean", "Symmetry Mean"],
    "Heart Disease": ["Age", "Sex (0=Female, 1=Male)", "Chest Pain Type", "Resting Blood Pressure", "Cholesterol", "Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", "Resting ECG", "Max Heart Rate", "Exercise-Induced Angina (1=Yes, 0=No)"],
    "Hypertension": ["Systolic BP", "Diastolic BP", "Age", "BMI", "Smoking (1=Yes, 0=No)", "Alcohol Intake (1=Yes, 0=No)", "Physical Activity (1=Yes, 0=No)"]
}

input_values = []
for field in input_fields[selected_disease]:
    input_values.append(st.number_input(field, value=0.0))

if st.button("Predict"):
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scalers[selected_disease].transform(input_array)
    prediction = models[selected_disease].predict(input_scaled)
    result = "Positive" if prediction[0] == 1 else "Negative"
    st.success(f"Prediction: {result}")


