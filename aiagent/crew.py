
from crewai import Crew, Process 
from .agents import Researcher_Analyst, Analyst_expert
from .task import research_task, Analysis_task

crew = Crew(

    agents = [Researcher_Analyst, Analyst_expert],
    tasks = [research_task, Analysis_task],
    process = Process.sequential

)

result = crew.kickoff(inputs={'RUL':'145 CYCLES'})
print(result)