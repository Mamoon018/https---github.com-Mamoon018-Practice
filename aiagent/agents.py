import os 
from crewai import Agent 
from dotenv import load_dotenv
load_dotenv(dotenv_path='aiagent\.env')
from langchain_google_genai import ChatGoogleGenerativeAI

## now, we will call the gemini model

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
os.environ["GOOGLE_API_KEY"] = gemini_api_key

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             verbose=True,
                             temperature=0.5,
                             google_api_key = os.getenv("GEMINI_API_KEY"))


Researcher_Analyst = Agent(

    role = "Researcher Analyst",
    goal = "fetch the reasons why engines usually fail at {RUL} cycles",
    verbose = True, 
    memory = True, 
    backstory = (

        "you are a researcher who have ability to fetech the information similar to the user "
        "cases based on the {RUL} cycles, and come up with most common reasons engine fails "
 
    ),
    tools = [],
    llm = llm,
    allow_delegation = False

)

# Analyst Agent to provide recommending actions to improve engine RUL

Analyst_expert = Agent(

    role = "Senior Analyst",
    goal = "Recommend the mitigation strategies to increase the {RUL} which is remaining useful life of engine",
    verbose = True, 
    memory = True, 
    backstory = (

        "you are an engineer expert in airplane engines, "
        "you have capability to take the relevant sensor readings related to engine and remaining usability life cyles of any engine"
        "and based on that write the possible actions that can improve the remaining usability cycle of engine"

    ),
    tools = [],
    llm = llm,
    allow_delegation = False

)
