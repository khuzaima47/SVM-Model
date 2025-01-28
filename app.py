import streamlit as st
import joblib
import numpy as np

# Load the pre-trained model and scaler
model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler.joblib')

# Title of the application
st.title("SVM Prediction App")

# User input form
st.header("User Input Features")

# Collect user input
age = st.number_input("Age", min_value=18, max_value=100)
estimated_salary = st.number_input("Estimated Salary", min_value=15000, max_value=150000)
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert gender to numerical value
gender_mapping = {"Male": 0, "Female": 1}
gender_encoded = gender_mapping[gender]

# Prepare the input data
input_data = np.array([[gender_encoded, age, estimated_salary]])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 0:
        st.write("Prediction: **The User is unlikely to Purchase**")
    else:
        st.write("Prediction: **The User Is likely to Purchase**")