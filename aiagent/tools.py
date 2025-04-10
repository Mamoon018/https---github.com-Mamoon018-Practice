
import os 
from dotenv import load_dotenv
load_dotenv(dotenv_path='aiagent\.env')
from crewai_tools import SerperDevTool 

os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

## Initializing tool for internet searching capabilities
tool = SerperDevTool()
