import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv

# Add this import for the embedding function
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

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

# Define the embedding model name
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"

# MODIFIED: Use the Google Generative AI Embedding Function
# **Important**: The collection name and persist_directory MUST match train.py
agent_memory = ChromaAgentMemory(
    collection_name="chinook_memory",
    persist_directory="./vanna_chroma_db",
    embedding_function=GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_KEY,
        model_name=GOOGLE_EMBEDDING_MODEL
    )
)

class SimpleUserResolver(UserResolver):
    async def resolve_user(self, request_context: RequestContext) -> User:
        user_email = request_context.get_cookie('vanna_email') or 'guest@example.com'
        group = 'admin' if user_email == 'admin@example.com' else 'user'
        return User(id=user_email, email=user_email, group_memberships=[group])

user_resolver = SimpleUserResolver()

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