

from dotenv import load_dotenv
load_dotenv()
import os 

# we can giev serper api key if we want to do google search
os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

from crewai_tools import SerperDevTool

# initialize the tool for internet searching capabilities
tool = SerperDevTool()


