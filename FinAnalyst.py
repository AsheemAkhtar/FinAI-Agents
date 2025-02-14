from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.duckduckgo import DuckDuckGoTools
import openai
from dotenv import load_dotenv
from agno.playground import Playground, serve_playground_app
from agno.storage.agent.sqlite import SqliteAgentStorage
import os

agent_storage: str = "tmp/agents.db"

openai.api_key=os.getenv("OPENAI_API_KEY")
load_dotenv()


# WEB SEARCH AGENT

web_search_agent=Agent(
    name="web search agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    # Store the agent sessions in a sqlite database
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    show_tool_calls=True,
    markdown=True,
)

# FINANCE AGENT

finance_agent=Agent(
    name="AI Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    instructions=["Format your response using markdown and use tables to display data where possible."],
    # Store the agent sessions in a sqlite database
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    # Adds the current date and time to the instructions
    add_datetime_to_instructions=True,
    # Adds the history of the conversation to the messages
    add_history_to_messages=True,
    # Number of history responses to add to the messages
    num_history_responses=5,
    show_tool_calls=True,
    markdown=True,
)

# TO RUN THIS AGENT LOCALLY
multi_ai_agent=Agent(
    team=[web_search_agent,finance_agent],
    instructions=["Always include sources","Format your response using markdown and use tables to display data where possible."],
    show_tool_calls=True,
    markdown=True,
)

multi_ai_agent.print_response("summarize analyst recommendations and share latest news for tesla",stream=True)

# TO RUN THIS AGENT ON AGNO'S PLAYGROUNG

app = Playground(agents=[web_search_agent,finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)