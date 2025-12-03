import sqlite3 
import pandas as pd 
import os
import asyncio
from dotenv import load_dotenv

from vanna import Agent, AgentConfig 
from vanna.core.registry import ToolRegistry
from vanna.core.user import UserResolver, User, RequestContext
from vanna.servers.fastapi import VannaFastAPIServer
from vanna.tools import RunSqlTool, VisualizeDataTool
from vanna.tools.agent_memory import (
    SaveQuestionToolArgsTool, 
    SearchSavedCorrectToolUsesTool, 
    SaveTextMemoryTool
)
from vanna.integrations.chromadb import ChromaAgentMemory 
from vanna.integrations.google import GeminiLlmService
from vanna.integrations.sqlite import SqliteRunner

## ⚙️ Configuration

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    print("FATAL: GEMINI_API_KEY environment variable not set. Please check your .env file.")
    exit(1)

llm = GeminiLlmService(
    model="gemini-2.5-flash-lite",
    api_key=GEMINI_KEY
)

DB_PATH = "./Chinook.sqlite"
db_tool = RunSqlTool(
    sql_runner=SqliteRunner(database_path=DB_PATH))

agent_memory = ChromaAgentMemory(
    collection_name="chinook_memory",
    persist_directory="./vanna_chroma_db"
)

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

user_resolver = SimpleUserResolver()

## 🧠 Training (RAG Step)

print("--- Starting Vanna DDL Training ---")

async def run_ddl_training(db_path, agent_memory):
    """Asynchronously runs the DDL training loop."""
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL"
        df_ddl = pd.read_sql_query(query, conn)
        conn.close()

        if df_ddl.empty:
            print(f"Warning: No DDL found in {db_path}. Training skipped.")
            return
        
        print(f"DEBUG: DDL extracted. Total items to train: {len(df_ddl)}.")
        
        for ddl in df_ddl['sql'].to_list():
            await agent_memory.save_text_memory(ddl, 'DDL for Chinook database') 
        
        print(f"Successfully trained {len(df_ddl)} DDL items into ChromaDB.")

    except Exception as e:
        print(f"Error during DDL extraction/training: {e}")

asyncio.run(run_ddl_training(DB_PATH, agent_memory))
print("--- Training Complete ---")

## 🤖 Agent Setup

tools = ToolRegistry()
tools.register_local_tool(db_tool, access_groups=['admin', 'user'])
tools.register_local_tool(SaveQuestionToolArgsTool(), access_groups=['admin']) 
tools.register_local_tool(SearchSavedCorrectToolUsesTool(), access_groups=['admin', 'user']) 
tools.register_local_tool(SaveTextMemoryTool(), access_groups=['admin', 'user']) 
tools.register_local_tool(VisualizeDataTool(), access_groups=['admin', 'user'])

agent = Agent(
    llm_service=llm,
    tool_registry=tools,
    user_resolver=user_resolver,
    agent_memory=agent_memory,
    config=AgentConfig() 
)

## 🚀 Run the Server

server = VannaFastAPIServer(agent)
server.run()