import pandas as pd
import os
from dotenv import load_dotenv

# Add this import for the embedding function (KEEPING GEMINI EMBEDDING)
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

# --- LLM SERVICE CHANGE: CORRECT IMPORT AND CLASS NAME ---
# Using the exact import path and class name you verified: OllamaLlmService
from vanna.integrations.ollama import OllamaLlmService
# ---

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
# from vanna.integrations.google import GeminiLlmService # REMOVED: Replaced by Ollama
from vanna.integrations.sqlite import SqliteRunner

## ⚙️ Configuration

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    # CRITICAL: The GEMINI_KEY is still required for the embedding_function!
    print("FATAL: GEMINI_API_KEY environment variable not set. It is required for the Google Generative AI Embedding Function.")
    exit(1)

# --- LLM SERVICE INITIALIZATION: Replaced GeminiLlmService with OllamaLlmService ---
llm = OllamaLlmService(
    model="llama3-groq-tool-use", 
    host=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434") 
)
# ---

DB_PATH = "./Chinook.sqlite"
db_tool = RunSqlTool(
    sql_runner=SqliteRunner(database_path=DB_PATH))

# Define the embedding model name
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"

# --- AGENT MEMORY: This section remains UNCHANGED as requested ---
agent_memory = ChromaAgentMemory(
    collection_name="chinook_memory",
    persist_directory="./vanna_chroma_db",
    embedding_function=GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_KEY, 
        model_name=GOOGLE_EMBEDDING_MODEL
    )
)
# ---

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