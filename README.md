
-- Demo Video: https://www.linkedin.com/feed/update/urn:li:activity:7316461739926069248/

# RUL Prediction Model & AI recommendation system

## Overview
Time series data of sensor readings of multiple cycles of engines was used to predict the Remaining Useful Cycle (RUL). Project takes the input in the form of different sensor readings & predicts the RUL of engine.  
The `agents.py` file defines two AI agents, `Researcher_Analyst` and `Analyst_expert`, which are designed to analyze engine failure reasons and recommend strategies to improve the Remaining Useful Life (RUL) of engines. These agents utilize the Google Gemini-Pro language model for advanced natural language processing and are configured to perform specific tasks with precision and efficiency.

## Dependencies
This module relies on the following:
- **`dotenv`**: For loading environment variables from the `.env` file
- **Data Handling**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn`, `KNNImputer`
- **Custom Modules**: `src.exception`, `src.logger`, `src.utils`
- **File Handling**: `os`, `pickle`
- **Logging and Error Handling**: `logging`, `sys`
- **`crewai`**: For creating and managing AI agents
- **`langchain_google_genai`**: To integrate with the Google Gemini-Pro language model

## Data Ingestion
   - Reads the raw data using `pandas`.
   - Splits the data into training and testing datasets.
   - Saves the processed data as CSV files in the `artifacts` directory:
     - `artifacts/train.csv`
     - `artifacts/test.csv`
## Data Transformation
- Feature Engineering
- Outlier Removal
- Missing Value Imputation
- Scaling
- Pipeline

## Model Trainer
1. **Models**:
   - CNN-LSTM (Keras)
   - K-Neighbors Regressor
   - XGBoost Regressor
   - CatBoost Regressor
   - AdaBoost Regressor
   - Gradient Boosting Regressor
   - Random Forest Regressor
   - Decision Tree Regressor


## AI Recommendation System

## Tools
Currently, Web searcher tool SERPER SEARCH is assigned to the agents. The framework supports tool integration, such as web search or data retrieval tools, which enhances the agents' capabilities.

## Agents
### 1. `Researcher_Analyst`
- **Role**: Researcher Analyst.
- **Goal**: Fetch reasons why engines typically fail at specific RUL cycles.
- **Features**:
  - Verbose output for detailed responses.
  - Memory-enabled for context retention.
  - Usage of external tool.
- **Backstory**: A researcher skilled in gathering information related to engine failures based on RUL cycles from web search.

### 2. `Analyst_expert`
- **Role**: Senior Analyst.
- **Goal**: Recommend strategies to improve the RUL of engines based on sensor data.
- **Features**:
  - Verbose output for detailed recommendations.
  - Memory-enabled for context retention.
  - Analysis of the collected data of web search to analyze and recommend mitigation strategies.
- **Backstory**: An expert in airplane engines, capable of analyzing sensor data to suggest actionable improvements.

## Tasks
The agents are designed to perform the following tasks:
1. **Research Task**:
   - **Agent**: `Researcher_Analyst`.
   - **Description**: Identify common reasons for engine failure at a given RUL cycle.
   - **Output**: A list of at least 5 failure reasons.

2. **Analysis Task**:
   - **Agent**: `Analyst_expert`.
   - **Description**: Recommend actions to improve engine RUL based on sensor data.
   - **Output**: At least 3 actionable recommendations.

These tasks are executed sequentially to provide a comprehensive analysis of engine performance and improvement strategies.

**Summary**:
AI system, defining specialized agents to address engine failure analysis and RUL improvement. By leveraging the Gemini-Pro language model, these agents deliver precise and actionable insights. The modular design allows for easy integration of additional tools and tasks, making it a flexible and scalable solution for engine diagnostics and optimization.
