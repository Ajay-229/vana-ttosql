# All imports at the top
import sqlite3 
import pandas as pd 

import asyncio

from vanna import Agent, AgentConfig 
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.tools import RunSqlTool, VisualizeDataTool
from vanna.tools.agent_memory import SaveQuestionToolArgsTool, SearchSavedCorrectToolUsesTool, SaveTextMemoryTool
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.integrations.google import GeminiLlmService
from vanna.integrations.sqlite import SqliteRunner
from vanna.integrations.local.agent_memory import DemoAgentMemory


# --- Configuration ---

# Configure your LLM
llm = GeminiLlmService(
    model="gemini-2.5-flash-lite",  # Using the free tier model
    api_key="AIzaSyCs9A2MIOL5JL2TF4V8-jPql1DIMB-DQvY"  # Your specific key
)

# Configure your database
db_tool = RunSqlTool(
    sql_runner=SqliteRunner(database_path="./Chinook.sqlite")) # Using the local sample DB

# Configure your agent memory
agent_memory = DemoAgentMemory(max_items=1000)

# Configure user authentication (simple cookie resolver)
class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

user_resolver = SimpleUserResolver()

# --- Training (RAG Step) - THE ABSOLUTE FINAL FIX ---

# --- Training (RAG Step) - FINAL AWAIT FIX ---

print("--- Starting Vanna Training ---")
# ... (DEBUG prints unchanged) ...

async def run_ddl_training(df_ddl, agent_memory):
    """Asynchronously runs the DDL training loop."""
    for ddl in df_ddl['sql'].to_list():
        # *** FIX: Using await for the async method ***
        # The save_text_memory method is a coroutine and must be awaited.
        await agent_memory.save_text_memory(ddl, 'DDL for Chinook database') 
        print(f"Trained DDL on Agent Memory: {ddl[:50]}...")

try:
    # 1. Use standard sqlite3 and pandas to safely extract DDL 
    conn = sqlite3.connect("./Chinook.sqlite")
    query = "SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL"
    df_ddl = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"DEBUG: DDL extracted. Total items to train: {len(df_ddl)}.")
    
    # 2. Run the asynchronous training loop synchronously
    asyncio.run(run_ddl_training(df_ddl, agent_memory))
    
except Exception as e:
    print(f"Error during DDL extraction/training: {e}")

print("--- Training Complete ---")
# --- Agent Creation ---

# Create your tools registry and register all necessary tools
tools = ToolRegistry()
# 1. Database Query Tool 
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
# 2. Agent Memory Tools 
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin'])
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user'])
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user'])
# 3. Data Visualization Tool
tools.register_local_tool(VisualizeDataTool(), access_groups=['admin', 'user'])

# Create your agent
agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    agent_memory=agent_memory,
    config=AgentConfig() 
)

# --- Run the Server ---
server = VannaFastAPIServer(agent)
server.run()