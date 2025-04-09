
from crewai import Agent
from .tools import tool 
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os 


class CustomGoogleLLM:
    def __init__(self):
        self.model = "gemini-pro"  # Using gemini-pro model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables!")
        print(f"Using API Key: {api_key[:5]}... (first 5 chars)")  # Debug output
        print(f"Model: {self.model}")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            verbose=True,
            temperature=0.5,
            google_api_key=os.getenv("GEMINI_API_KEY")
        )

    def call(self, messages, stop = None , **kwargs):

        from langchain_core.messages import HumanMessage, AIMessage
        # Convert CrewAI messages to LangChain format
        langchain_messages = []
        for m in messages:
            if m["role"] == "user":
                langchain_messages.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                langchain_messages.append(AIMessage(content=m["content"]))

        response = self.llm.invoke(langchain_messages)
        return response.content

llm = CustomGoogleLLM()


Researcher_Analyst = Agent(

    role = "Researcher Analyst",
    goal = "fetch the reasons why engines usually fail at {RUL} cycles",
    verbose = True, 
    memory = True, 
    backstory = (

        "you are a researcher who have ability to fetech the information similar to the user "
        "cases based on the {RUL} cycles, and come up with most common reasons engine fails "
 
    ),
    tools = [tool],
    llm = llm,
    allow_delegation = True

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

