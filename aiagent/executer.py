

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
    # Get individual task outputs
    research_output = research_task.output.raw if research_task.output else "No output from research task."
    analysis_output = Analysis_task.output.raw if Analysis_task.output else "No output from analysis task."
    
    # Return a dictionary with both outputs
    return {
        "research_output": research_output,
        "analysis_output": analysis_output
    }


