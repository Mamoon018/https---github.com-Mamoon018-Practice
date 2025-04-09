# TimeseriesProject

            ## Data_Ingestion.py ## 
# Overview
The data_ingestion.py file is responsible for the initial step of the data science project pipeline: loading and preparing the raw datasets. It reads the training and test data from text files, processes them into a structured format (CSV), and saves them into the artifacts directory for downstream use. This module ensures that the data is ingested correctly and made available for subsequent transformation and modeling steps.

# Key Components
Configuration Class: dataingestioncongif
A dataclass that defines file paths for storing the processed train and test datasets.

# Default paths:
Train data: artifacts/train.csv
Test data: artifacts/test.csv

# Main Class: dataingestion
Initializes with an instance of dataingestioncongif to manage file paths.
Contains the core method initiate_data_ingestion() for data loading and saving.

# Method: initiate_data_ingestion()

Input: Reads raw data from Notebook/Data/train_FD001.txt and Notebook/Data/test_FD001.txt using pandas, assuming whitespace-delimited text files without headers.

Process:Creates the artifacts directory if it doesn’t exist.
Saves the train and test datasets as CSV files with headers in the specified paths.

Output: Returns the file paths of the saved train and test CSV files.

Notes: The code assumes pre-split train and test datasets, so no additional train-test splitting is performed (commented-out section).


# Error Handling
Utilizes a custom exception class (customException) to handle and log errors during ingestion.
Logs key steps and exceptions using the logging module from src.logger.

# Execution Block
When run as a standalone script (if __name__ == "__main__":), it:
1: Instantiates the dataingestion class.

2: Executes the ingestion process.

3: Passes the resulting train and test data paths to the data_transformation and model_trainer components for further processing.

        ## Data_Transformation.py ## 

# Overview
The data_transformation.py file handles data preprocessing and feature engineering for the project. It transforms raw train and test datasets by generating new features, removing outliers, imputing missing values, and scaling numerical columns. The processed data is returned as NumPy arrays, and the preprocessing object is saved for later use.

# Key Components
# Configuration Class: datatransformationconfig
Defines the path for saving the preprocessing object: artifacts/preprocessing.pkl.

# Custom Transformers
feature_engineering: Adds features like Remaining Useful Life (RUL), rolling statistics, and delta columns based on sensor data.
sd_outlier_removal: Removes outliers in specified sensor columns using standard deviation (3σ rule).
IQR_outlier_removal: Removes outliers in select sensor columns using the Interquartile Range (IQR) method.
imputing_Nan_using_KNN: Imputes missing values in numerical columns using KNN with highly correlated features.

# Main Class: data_transformation
get_datapreprocessing(): Builds a preprocessing pipeline with outlier removal, KNN imputation, and scaling for numerical columns.
transform_data(train_data, test_data):
Reads train and test CSV files.
Applies feature engineering and preprocessing.
Returns transformed train and test arrays with RUL as the target, and saves the preprocessor.

# Dependencies
numpy, pandas: Data manipulation.
sklearn: Pipelines, transformers, imputation, and scaling.
Custom modules: src.exception, src.logger, src.utils.

# Usage
Takes train and test data paths as input.
Outputs:
train_arr: Transformed training data with RUL.
test_arr: Transformed test data (last cycle per unit).
Preprocessor saved at artifacts/preprocessing.pkl.


        ## Model_trainer.py ##

# Overview
The model_trainer.py file manages the training, evaluation, and selection of machine learning models for the project. It trains multiple regression models, tunes their hyperparameters, evaluates performance using R² scores, and saves the best model for future use.

# Key Components

# Configuration Class: ModelTrainerConfig
Defines the path for saving the trained model: artifacts/model.pkl.

# Main Class: ModelTrainer
initiaing_model_trainer(train_array, test_array, rul):
Splits input arrays into features (X_train, X_test) and targets (Y_train, Y_test using RUL).
Defines a set of regression models and their hyperparameter grids.
Evaluates models using the evaluate_models utility and selects the best performer based on R² score.
Saves the best model and returns its accuracy, name, score, instance, and full report.

# Models and Tuning
Models: CNN-LSTM (Keras), K-Neighbors, XGBoost, CatBoost, AdaBoost, Gradient Boosting, Random Forest, Decision Tree.
Hyperparameters: Defined in tuning_parameters (partially commented out for brevity in execution).

# Dependencies
sklearn, catboost, xgboost, tensorflow: For model implementations.
Custom modules: src.exception, src.logger, src.utils.
scikeras.wrappers: For Keras integration.

# Usage
Takes transformed train and test arrays, plus RUL values, as input.

# Outputs:
R² accuracy of the best model on test data.
Best model name, score, instance, and a report of all models.

Saves the best model at artifacts/model.pkl.


        ## tools.py ##

# Overview
The tools.py file sets up an external web search tool using SerperDevTool from crewai_tools, enabling agents to fetch real-time data from the internet to support their tasks, such as researching engine failure reasons.

# Key Components

# Tool Initialization:
Loads the SERPER_API_KEY from a .env file using dotenv.
Configures SerperDevTool as tool, a reusable search utility for agents.
# Environment Setup: 
Assigns the API key to os.environ for tool authentication.

# Dependencies
dotenv: Loads environment variables.
os: Manages environment settings.
crewai_tools: Provides SerperDevTool for web searches.

# Usage
Exported as tool for use in agents.py and task.py.
Example: An agent can call tool to search for "engine failure reasons at 145 cycles."


            ### AI AGENTS PART ###

# Overview
The agents.py file defines two AI agents using the crewai framework, powered by the Google Gemini-Pro language model. These agents are designed to analyze engine failure reasons and recommend strategies to improve Remaining Useful Life (RUL) based on sensor data and research insights.

# Key Components

# Custom LLM: CustomGoogleLLM
Wraps the ChatGoogleGenerativeAI model (gemini-pro) for integration with crewai.
Uses an API key from environment variables (GEMINI_API_KEY).
Converts crewai message formats to LangChain-compatible messages for processing.

# Agent 1: Researcher_Analyst
    Role: Researcher Analyst.
    
    Goal: Identify common reasons for engine failure at a specified RUL cycle count.
    
    Features: Verbose output, memory-enabled, uses a custom tool (tool), and allows delegation.
    
    Backstory: A researcher skilled in fetching RUL-related failure data.

# Agent 2: Analyst_expert
    Role: Senior Analyst.
    
    Goal: Recommend mitigation strategies to extend engine RUL.
    
    Features: Verbose output, memory-enabled, no tools, and no delegation.
    
    Backstory: An airplane engine expert leveraging sensor data to suggest RUL improvements.

# Dependencies
crewai: For agent creation.
langchain_google_genai: For the Gemini-Pro LLM.
dotenv: For loading environment variables.
Custom module: .tools (assumed to provide tool).

# Usage
Requires a GEMINI_API_KEY set in a .env file.
Agents can be instantiated and used within a crewai workflow to process RUL-related tasks.


        ## Task.py ##

# Overview
The task.py file specifies two tasks for the agents using crewai. These tasks define the objectives—researching engine failure reasons and recommending RUL improvements—along with expected outputs and agent assignments.

# Key Components

# Task 1: research_task:
    Description: Research common engine failure reasons at a given {RUL} cycles.
    
    Expected Output: At least 5 failure reasons.
    
    Agent: Researcher_Analyst.
    
    Tools: Uses tool for web searches.

# Task 2: Analysis_task:
    Description: Recommend actions to improve engine RUL (set at {RUL}) based on sensor data.
    
    Expected Output: At least 3 actionable recommendations.
    Agent: Analyst_expert.
    
    Features: No tools, synchronous execution, outputs to Analysis_output.md.

# Dependencies
crewai: Task framework.
.tools: Provides tool for research_task.
.agents: Imports Researcher_Analyst and Analyst_expert.

# Usage
Tasks are assigned to agents and executed via crew.py or executer.py.
Example: research_task outputs failure reasons; Analysis_task writes recommendations to a file.


        ## Crew.py ##

# Overview
The crew.py file orchestrates the agents and tasks into a sequential workflow using crewai. It executes the research and analysis tasks with a hardcoded RUL input and prints the combined result.

# Key Components

# Crew Setup:
Agents: Researcher_Analyst and Analyst_expert.
Tasks: research_task and Analysis_task.
Process: Process.sequential ensures tasks run one after another.

# Execution:
Runs the crew with inputs={'RUL': '145'}.
Outputs the combined result of both tasks.

# Dependencies
crewai: Crew and process management.
.task: Imports research_task and Analysis_task.
.agents: Imports Researcher_Analyst and Analyst_expert.


        ## Executer.py ##

# Overview
The executer.py file provides a reusable function to run the crewai workflow with a dynamic RUL input. It enhances flexibility over crew.py by returning individual task outputs in a structured format.

# Key Components

# Function: 
    run_agent_crew(rul_value):

1) Creates a crew with Researcher_Analyst, Analyst_expert, research_task, and Analysis_task.
2) Executes tasks sequentially using Process.sequential.
3) Extracts raw outputs from each task, handling cases where outputs are missing.
4) Returns a dictionary with research_output and analysis_output.

# Dependencies
crewai: Crew and process management.
.task: Imports research_task and Analysis_task.
.agents: Imports Researcher_Analyst and Analyst_expert.