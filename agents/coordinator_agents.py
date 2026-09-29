from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage
import os

from graph.state import AgentState

# llm = ChatOllama(
#     model="qwen3:4b"
# )


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=api_key
)

COORDINATOR_PROMPT = """
You are the Coordinator Agent of a multi-agent database system.

Your job is ONLY to determine which agent should handle the user's request.

Available agents:

1. sql_agent
   - Used for database operations.
   - SELECT
   - INSERT
   - UPDATE
   - DELETE
   - CREATE TABLE
   - DROP TABLE
   - schema inspection
   - SQL queries
   - database-related questions

2. quality_agent
   - Used for checking data quality problems.
   - Examples:
     - missing values
     - duplicate records
     - invalid values
     - inconsistent data
     - suspicious data

IMPORTANT:
- Do NOT execute SQL.
- Do NOT answer the user's database question.
- ONLY select the appropriate agent.
- If the request is a normal database operation or database query, select sql_agent.
- If the request explicitly asks to analyze data quality, select quality_agent.

Return ONLY one of:

sql_agent
quality_agent
"""

def coordinatorAgent(state: AgentState):
    messages = [
        SystemMessage(content=COORDINATOR_PROMPT)
    ] + state["messages"]

    response = llm.invoke(messages)

    decision = response.content.strip().lower()

    if "quality_agent" in decision:
        selected_agent = "quality_agent"
    else:
        selected_agent = "sql_agent"

    print("\n[COORDINATOR]")
    print(f"Selected agent: {selected_agent}")

    return {
        "selected_agent": selected_agent
    }