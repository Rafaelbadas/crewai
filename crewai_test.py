import os
from langchain_google_genai import ChatGoogleGenerativeAI
from crewai import Agent, Task, Crew, Process
from IPython.display import Markdown
import streamlit as st

# Set gemini pro as llm
llm = ChatGoogleGenerativeAI(model="gemini-pro",verbose = True,temperature = 0.6,
                             google_api_key="API_KEY")
from langchain.tools import DuckDuckGoSearchRun
search_tool = DuckDuckGoSearchRun()
researcher = Agent(
  role='Senior Research Analyst',
  goal='Uncover cutting-edge developments in AI and data science',
  backstory="""You work at a leading tech think tank.
  Your expertise lies in identifying emerging trends.
  You have a knack for dissecting complex data and presenting
  actionable insights.""",
  verbose=True,
  max_iter = 60,
  allow_delegation=False,
  llm = llm,  #using google gemini pro API
  tools=[
        search_tool
      ]
)
writer = Agent(
  role='Tech Content Strategist',
  goal='Craft compelling content on tech advancements',
  backstory="""You are a renowned Content Strategist, known for
  your insightful and engaging articles.
  You transform complex concepts into compelling narratives.""",
  verbose=True,
  max_iter = 60,
  allow_delegation=False,
  llm = llm,  #using google gemini pro API
  tools=[]
)
# Create tasks for your agents
task1 = Task(
  description="""Conduct a comprehensive analysis of the latest advancements in AI in 2024.
  Identify key trends, breakthrough technologies, and potential industry impacts.""",
  expected_output="""Your final answer MUST be a full analysis report""",
  agent=researcher
)


task2 = Task(
  description="""Using the insights provided, develop an engaging blog
  post that highlights the most significant AI advancements.""",
  expected_output=""""Your post should be informative yet accessible, catering to a tech-savvy audience.
  Make it sound cool, avoid complex words so it doesn't sound like AI.
  Your final answer MUST be the full blog post of at least 4 paragraphs.
  """,
  agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
  agents=[researcher, writer],
  tasks=[task1, task2],
  full_output = True,
  max_iter = 60,
  verbose=1, # You can set it to 1 or 2 to different logging levels
)
result = crew.kickoff()
#result.raw
with st.sidebar:
    st.header('Enter the ai header')

    with st.form(key='research_form'):
        topic = st.text_input("Select the ai topic")
        submit_button = st.form_submit_button(label = "Run Research")

if submit_button:
    if not topic:
        st.error("Please fill the ai field")
    else:
        #result= crew.kickoff()

        st.subheader("Results of your research:")
        st.write(result.raw)
