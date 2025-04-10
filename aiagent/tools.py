
import os 
from dotenv import load_dotenv
load_dotenv(dotenv_path='aiagent\.env')


os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

from crewai_tools import SerperDevTool

## Initializing tool for internet searching capabilities
tool = SerperDevTool()
