from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from graph.state import AgentState

from agents.coordinator_agents import coordinatorAgent
from agents.sql_agents import sql_agent, sql_tool_node


graph_builder = StateGraph(AgentState)

graph_builder.add_node(
    "coordinator",
    coordinatorAgent
)

graph_builder.add_node(
    "sql_agent",
    sql_agent
)

graph_builder.add_node(
    "sql_tools",
    sql_tool_node
)

graph_builder.add_edge(
    START,
    "coordinator"
)

def route_from_coordinator(state: AgentState):

    selected_agent = state["selected_agent"]

    if selected_agent == "sql_agent":
        return "sql_agent"

    if selected_agent == "quality_agent":
        # Quality Agent belum dibuat.
        # Untuk sementara kita hentikan di sini.
        return END

    return END


graph_builder.add_conditional_edges(
    "coordinator",
    route_from_coordinator
)

graph_builder.add_conditional_edges(
    "sql_agent",
    tools_condition,
    {
        "tools": "sql_tools",
        "__end__": END
    }
)

graph_builder.add_edge(
    "sql_tools",
    "sql_agent"
)

memory = InMemorySaver()

graph = graph_builder.compile(
    checkpointer=memory
)