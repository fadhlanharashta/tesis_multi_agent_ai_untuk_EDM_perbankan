from langgraph.graph import MessagesState
from typing import Literal

class AgentState(MessagesState):
    selected_agent: Literal[
        "sql_agent",
        "quality_agent"
    ]