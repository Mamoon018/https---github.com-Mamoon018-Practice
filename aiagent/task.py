
from crewai import Task 
from .agents import Researcher_Analyst, Analyst_expert

# Research Task 
research_task = Task(

    description= (
        "you are a researcher who have ability to fetech the information similar to the user "
        "cases based on the {RUL} cycles, and come up with most common reasons engine fails "
    ),
    expected_output= "At least 5 reasons related to the engine failure after {RUL} cycles",
    tools= [],
    agent = Researcher_Analyst,

)

# Analysis Task 
Analysis_task = Task(

    description=(

        "you are an engineer expert in airplane engines, "
        "you have capability to take the relevant sensor readings related to engine and remaining usability life cyles of any engine"
        "remaining useability is {RUL} and now based on that write the possible actions that can improve the remaining usability cycle of engine"
    ),
    expected_output= "Provide at least 3 points of recommendation for best possible actions that can improve the remaining usability cycyle of engine",
    tools= [],
    agent= Analyst_expert,
    async_execution= False,
    output_file= 'Analysis_output.md' # Example of output customization

)