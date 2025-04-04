
from crewai import Crew, Process 
from task import research_task, Analysis_task
from agents import Researcher_Analyst, Analyst_expert

## Forming the tech focused crew with some enhanced configuration
crew=Crew(
    agents=[Researcher_Analyst,Analyst_expert],
    tasks=[research_task, Analysis_task],
    process=Process.sequential,

)

## starting the task execution process

result=crew.kickoff(inputs={'RUL':'145'})
print(result)

