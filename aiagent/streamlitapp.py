
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import sys 
import os 

project_root = "C:/Users/Hp/Projects/Timeseriesproject"
sys.path.append(project_root)

st.title("Predictive Maintenance for Manufacturing Equipment")
st.write("Predict the Remaining Useful Life (RUL) of engines")

# Load the trained model
model = joblib.load(os.path.join(project_root, "artifacts", "model.pkl"))
preprocessor = joblib.load(os.path.join(project_root, "artifacts", "preprocessing.pkl"))


# Load the training data to calculate mean values
df = pd.read_csv('C:/Users/Hp/Projects/Timeseriesproject/Notebook/artifacts/data.csv')


column_names = ['unit_number', 'times_in_cycles', 'ops_1', 'ops_2', 'ops_3'] + [f'sensor_{i}' for i in range(1, 22)]
df.columns = column_names + [col for col in df.columns if col not in column_names]  # Adjust for derived features

mean_values = df.mean()

# Prediction Input
st.header("Prediction Input")
times_in_cycles = st.number_input("times_in_cycles", min_value=0.0, max_value=362.0, value=0.0000, step=0.0001, format="%.4f")
ops_1 = st.number_input("ops_1", min_value=-0.0087, max_value=0.0087, value=0.0000, step=0.0001, format="%.4f")
ops_2 = st.number_input("ops_2", min_value= -0.0005, max_value=0.0006, value=0.0000, step=0.0001, format="%.4f")
sensor_4 = st.number_input("sensor_4", min_value=1382.25, max_value= 1435.9, value=1400.0, step=0.0001, format="%.4f")
sensor_9 = st.number_input("sensor_9", min_value=0.0, max_value=100.0, value=50.0, step=0.0001, format="%.4f")
sensor_17 = st.number_input("sensor_17", min_value=0.0, max_value=100.0, value=50.0, step=0.0001, format="%.4f")


if st.button("Predict RUL"):
    # Create a DataFrame with user inputs and default values
    input_data = pd.DataFrame([mean_values], columns=df.columns)
    input_data['times_in_cycles'] = times_in_cycles
    input_data['ops_1'] = ops_1
    input_data['ops_2'] = ops_2
    input_data['sensor_4'] = sensor_4
    input_data['sensor_9'] = sensor_9
    input_data['sensor_17'] = sensor_17

    # Drop RUL column if present
    if 'RUL' in input_data.columns:
        input_data = input_data.drop(columns=['RUL'])

    # Apply the preprocessor
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_processed)
    st.success(f"Predicted RUL: {prediction[0]:.2f} cycles")
