import sys
import os

from aiagent.executer import run_agent_crew

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
from src.logger import logging


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
    # Createing a DataFrame with user inputs and default values
    input_data = pd.DataFrame([mean_values], columns=df.columns)
    input_data['times_in_cycles'] = times_in_cycles
    input_data['ops_1'] = ops_1
    input_data['ops_2'] = ops_2
    input_data['sensor_4'] = sensor_4
    input_data['sensor_9'] = sensor_9
    input_data['sensor_17'] = sensor_17

    if 'RUL' in input_data.columns:
        input_data = input_data.drop(columns=['RUL'])

    # Apply the preprocessor
    input_processed = preprocessor.transform(input_data)

# Make prediction
    prediction = model.predict(input_processed)
    st.session_state.predicted_rul = prediction[0]  # Store prediction in session state
    st.success(f"Predicted RUL: {prediction[0]:.2f} cycles")
    logging.info(f"Predicted RUL: {prediction[0]:.2f}")

# AI Agent Analysis Section
if st.button("Show Analysis", key="analysis_button"):
    if 'predicted_rul' in st.session_state:
        st.success(f"Predicted RUL: {st.session_state.predicted_rul:.2f} cycles")
        st.header("2. AI Agent Analysis")
        st.write(f"Run analysis for the predicted RUL of {st.session_state.predicted_rul:.2f} cycles?")
        
        # Checkbox to trigger agent analysis
        run_agents = st.checkbox("Get Analysis by AI Agents", key="run_agents_checkbox")
        
        if run_agents:
            with st.spinner("Running AI agents for failure analysis and mitigation strategies..."):
                try:
                    # Log and display the input
                    rul_input = str(round(st.session_state.predicted_rul))
                    logging.info(f"Calling run_agent_crew with input: {rul_input}")
                    st.write(f"Debug: Calling run_agent_crew with input: {rul_input}")
                    
                    # Run the agent crew
                    agent_results = run_agent_crew(rul_input)
                    logging.info(f"run_agent_crew returned: {agent_results}")
                    
                    # Debug: Show raw output
                    st.write(f"Debug: Raw agent_results: {agent_results}")
                    st.write(f"Debug: Type of agent_results: {type(agent_results)}")
                    
                    # Extract and display individual outputs
                    research_output = agent_results.get("research_output", "No research output available.")
                    analysis_output = agent_results.get("analysis_output", "No analysis output available.")
                    
                    if research_output == "No output from research task." and analysis_output == "No output from analysis task.":
                        st.warning("No meaningful output received from AI agents.")
                        logging.warning("Both agent outputs are empty.")
                    else:
                        # Display research output
                        st.subheader("Research Analyst Insights")
                        st.markdown("**Reasons for Engine Failure:**")
                        st.markdown(research_output)
                        
                        # Display analysis output
                        st.subheader("Senior Analyst Recommendations")
                        st.markdown("**Mitigation Strategies:**")
                        st.markdown(analysis_output)
                except Exception as e:
                    st.error(f"Error running AI agents: {str(e)}")
                    logging.error(f"Exception in run_agent_crew: {str(e)}")
    else:
        st.warning("Please predict RUL first.")