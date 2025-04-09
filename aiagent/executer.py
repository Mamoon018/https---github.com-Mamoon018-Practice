

from crewai import Crew, Process
from .task import research_task, Analysis_task
from .agents import Researcher_Analyst, Analyst_expert

def run_agent_crew(rul_value: str):
    crew = Crew(
        agents=[Researcher_Analyst, Analyst_expert],
        tasks=[research_task, Analysis_task],
        process=Process.sequential,
    )
    result = crew.kickoff(inputs={'RUL': rul_value})
    return result


