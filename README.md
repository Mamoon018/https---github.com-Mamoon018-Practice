# AI Agents Module

## Overview
The `agents.py` file defines two AI agents, `Researcher_Analyst` and `Analyst_expert`, which are designed to analyze engine failure reasons and recommend strategies to improve the Remaining Useful Life (RUL) of engines. These agents utilize the Google Gemini-Pro language model for advanced natural language processing and are configured to perform specific tasks with precision and efficiency.

## Dependencies
This module relies on the following:
- **`crewai`**: For creating and managing AI agents.
- **`langchain_google_genai`**: To integrate with the Google Gemini-Pro language model.
- **`dotenv`**: For loading environment variables from the `.env` file.

Ensure the `.env` file contains the `GEMINI_API_KEY` for accessing the Gemini-Pro model.

## Tools
Currently, no external tools are assigned to the agents. However, the framework supports tool integration, such as web search or data retrieval tools, which can be added to enhance the agents' capabilities.

## Agents
### 1. `Researcher_Analyst`
- **Role**: Researcher Analyst.
- **Goal**: Fetch reasons why engines typically fail at specific RUL cycles.
- **Features**:
  - Verbose output for detailed responses.
  - Memory-enabled for context retention.
  - No delegation or external tools.
- **Backstory**: A researcher skilled in gathering information related to engine failures based on RUL cycles.

### 2. `Analyst_expert`
- **Role**: Senior Analyst.
- **Goal**: Recommend strategies to improve the RUL of engines based on sensor data.
- **Features**:
  - Verbose output for detailed recommendations.
  - Memory-enabled for context retention.
  - No delegation or external tools.
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

## Summary
The `agents.py` module is a critical part of the AI system, defining specialized agents to address engine failure analysis and RUL improvement. By leveraging the Gemini-Pro language model, these agents deliver precise and actionable insights. The modular design allows for easy integration of additional tools and tasks, making it a flexible and scalable solution for engine diagnostics and optimization.
