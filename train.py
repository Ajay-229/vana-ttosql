import sqlite3
import pandas as pd
import os
import asyncio
from dotenv import load_dotenv
import shutil
import textwrap # NEW: Helpful for formatting document content

# Vanna and ChromaDB imports
from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction
from vanna.integrations.chromadb import ChromaAgentMemory
from vanna.integrations.google import GeminiLlmService 

# --- Configuration ---
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    print("FATAL: GEMINI_API_KEY environment variable not set. Please check your .env file.")
    exit(1)

# Configuration must match app.py
DB_PATH = "./Chinook.sqlite"
COLLECTION_NAME = "chinook_memory" 
PERSIST_DIRECTORY = "./vanna_chroma_db"
GOOGLE_EMBEDDING_MODEL = "gemini-embedding-001"

# NEW: Document Path Configuration
# Set to an empty string if no document is used, or a path to a text file (e.g., 'business_rules.txt')
DOCUMENT_PATH = "./doc.txt" 

# --- Agent Memory Setup ---
agent_memory = ChromaAgentMemory(
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=GoogleGenerativeAiEmbeddingFunction(
        api_key=GEMINI_KEY,
        model_name=GOOGLE_EMBEDDING_MODEL
    )
)

# --- Utility Function ---

def is_memory_trained(persist_directory):
    """
    Checks if the ChromaDB persistent directory exists and contains data files.
    This serves as a reliable proxy for "memory trained."
    """
    if not os.path.exists(persist_directory):
        return False
    
    # Check for core ChromaDB files (like the chromadb.sqlite file)
    return any(os.path.isfile(os.path.join(persist_directory, f)) for f in os.listdir(persist_directory))

# --- Training Logic ---

async def train_ddl(db_path, agent_memory):
    """Extracts and trains DDL from the SQLite database."""
    conn = sqlite3.connect(db_path)
    query = "SELECT type, sql FROM sqlite_master WHERE sql IS NOT NULL"
    df_ddl = pd.read_sql_query(query, conn)
    conn.close()

    if df_ddl.empty:
        print(f"Warning: No DDL found in {db_path}. DDL training skipped.")
        return 0

    print(f"DEBUG: DDL extracted. Total items to train: {len(df_ddl)}.")
    
    for ddl in df_ddl['sql'].to_list():
        await agent_memory.save_text_memory(ddl, 'DDL for Chinook database')
        
    return len(df_ddl)

async def train_document(document_path, agent_memory):
    """Reads and trains a text document if the path is valid."""
    if not document_path:
        print("Document Path is empty. Skipping document training.")
        return 0
    
    if not os.path.exists(document_path):
        # Return an error if the path is provided but the file doesn't exist
        raise FileNotFoundError(f"Document file not found at: {document_path}")

    print(f"Processing document: {document_path}")
    
    # Simple approach: read the whole document as one chunk
    with open(document_path, 'r', encoding='utf-8') as f:
        document_content = f.read()

    # Save the document content as one large piece of context
    # Use textwrap to display a snippet of the saved content
    snippet = textwrap.shorten(document_content, width=80, placeholder="...")
    await agent_memory.save_text_memory(document_content, f'Business documentation from {document_path}')
    
    print(f"Successfully trained 1 document item: '{snippet}'")
    return 1


async def run_training_workflow(db_path, document_path, agent_memory, persist_directory):
    """Manages the full training and retraining workflow."""
    
    print("--- Starting Vanna Training Workflow ---")
    
    if is_memory_trained(persist_directory):
        print(f"Persistent memory found in '{persist_directory}'.")
        
        user_input = input("Trained memory exists. Do you want to DELETE and RETRAIN ALL data (DDL + Documents)? (yes/no): ").lower()
        if user_input != 'yes':
            print("Skipping training. Using existing memory.")
            return
        
        print(f"Deleting existing memory directory: {COLLECTION_NAME}...")
        try:
            shutil.rmtree(persist_directory) 
            print("Existing memory deleted. Starting retraining...")
        except Exception as e:
            print(f"Warning: Could not delete directory {persist_directory}. Proceeding anyway. Error: {e}")
            
    total_trained = 0
    try:
        # 1. Train DDL
        total_trained += await train_ddl(db_path, agent_memory)
        
        # 2. Train Document (if path is provided)
        total_trained += await train_document(document_path, agent_memory)

        print(f"\n✅ Training Complete! Total items trained: {total_trained}")

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
    except Exception as e:
        print(f"\nFATAL ERROR during training: {e}")


if __name__ == "__main__":
    # Call the main workflow function
    asyncio.run(run_training_workflow(DB_PATH, DOCUMENT_PATH, agent_memory, PERSIST_DIRECTORY))
    print("--- Vanna Training Workflow Finished ---")