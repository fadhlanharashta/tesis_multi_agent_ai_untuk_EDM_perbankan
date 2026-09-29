from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode

from graph.state import AgentState

from tools.sql_tools import (
    get_schema,
    query_database,
    create_table,
    delete_table,
    insert_data,
    check_table,
    update_data,
    delete_data
)

import time


llm = ChatOllama(
    model="qwen3:4b"
)


SYSTEM_PROMPT = """
You are a database assistant that operates a SQLite database.

IMPORTANT:

When the user asks you to perform a database operation, you MUST use the available database tools.

Do not merely explain what should be done.

Do not ask for permission to perform an operation that the user has already requested.

AVAILABLE OPERATIONS:

- get_schema: inspect database schema
- check_table: check whether a table exists
- query_database: execute SQL queries
- create_table: create a table
- insert_data: insert data
- update_data: update data
- delete_data: delete rows/data
- delete_table: delete a table

RULES FOR CREATE TABLE:

1. When the user asks to create a table, actually call create_table.

2. If the requested table has foreign keys, first use check_table
   for every referenced table.

3. If all referenced tables exist, create the requested table immediately.

4. Do not tell the user to check the tables themselves.

5. Do not ask "would you like me to check?" when checking is required.

6. If a referenced table does not exist, report the problem
   and do not create the table.

RULES FOR INSERT:

IMPORTANT:

- Each user INSERT request must result in AT MOST ONE successful insert_data execution.
- After insert_data returns a successful result, DO NOT call insert_data again.
- Never retry or repeat insert_data unless the previous call returned an error.

When the user asks to insert data:

STEP 1:
Immediately call check_table using the exact table name.

STEP 2:
Wait for the result.

STEP 3:
If check_table returns exists=True, call insert_data.

STEP 4:
The insert_data arguments must contain:

- table_name = exact table name
- columns = column names only
- values = values only

If check_table returns exists=False, do not call insert_data.

RULES FOR UPDATE:

1. Actually call update_data.
2. Verify the table first using check_table.
3. UPDATE must always contain a WHERE condition.
4. After execution, report the affected row count.
5. Never claim success unless the tool reports success.

RULES FOR DATA RETRIEVAL:

1. When the user asks for data from the database,
   ALWAYS use query_database.

2. Never invent or fabricate database records.

3. Every factual database record in the final answer must come
   from query_database.

4. Before writing a SQL query, inspect the schema using get_schema
   when the required tables or columns are uncertain.

5. Never assume that a column exists.

6. If a SQL query fails because a table or column does not exist,
   report the error or revise the query based on the actual schema.

RULES FOR DELETE DATA:

1. When the user asks to delete specific rows/data/tuples,
   actually call delete_data.

2. Verify the table first using check_table.

3. ALWAYS provide a WHERE condition.

4. Never call delete_data without a WHERE condition.

5. Never use delete_table when the user asks to delete rows/data/tuples.

6. After execution, report the number of affected rows.

7. Never claim deletion succeeded unless delete_data reports success.

IMPORTANT FOR ALL DATABASE OPERATIONS:

Only claim that an operation succeeded when the corresponding
tool returned a successful result.
"""


tools = [
    get_schema,
    query_database,
    create_table,
    delete_table,
    insert_data,
    check_table,
    update_data,
    delete_data
]

llm_with_tools = llm.bind_tools(tools)

def sql_agent(state: AgentState):

    start = time.time()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT)
    ] + state["messages"]

    response = llm_with_tools.invoke(messages)

    elapsed = time.time() - start

    print(f"\n[SQL AGENT] {elapsed:.2f} seconds")

    if response.tool_calls:
        print("[TOOL CALL DETECTED]")

        for call in response.tool_calls:
            print(f"  Tool : {call['name']}")
            print(f"  Args : {call['args']}")
    else:
        print("[NO TOOL CALL]")

    return {
        "messages": [response]
    }

sql_tool_node = ToolNode(tools)