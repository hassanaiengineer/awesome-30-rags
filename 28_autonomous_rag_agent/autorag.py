import streamlit as st
import nest_asyncio
from io import BytesIO
from agno.agent import Agent
from agno.document.reader.pdf_reader import PDFReader
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.storage.agent.postgres import PostgresAgentStorage

# Apply nest_asyncio to allow nested event loops, required for running async functions in Streamlit
nest_asyncio.apply()

from branding import apply_streamlit_branding

# Database connection string for PostgreSQL
DB_URL = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Function to set up the Assistant, utilizing caching for resource efficiency
@st.cache_resource
def setup_assistant(api_key: str) -> Agent:
    """Initializes and returns an AI Assistant agent with caching for efficiency.

    This function sets up an AI Assistant agent using the OpenAI GPT-4o-mini model 
    and configures it with a knowledge base, storage, and web search tools. The 
    assistant is designed to first search its knowledge base before querying the 
    internet, providing clear and concise answers.

    Args:
        api_key (str): The API key required to access the OpenAI services.

    Returns:
        Agent: An initialized Assistant agent configured with a language model, 
        knowledge base, storage, and additional tools for enhanced functionality."""
    llm = OpenAIChat(id="gpt-4o-mini", api_key=api_key)
    # Set up the Assistant with storage, knowledge base, and tools
    return Agent(
        id="auto_rag_agent",  # Name of the Assistant
        model=llm,  # Language model to be used
        storage=PostgresAgentStorage(table_name="auto_rag_storage", db_url=DB_URL),  
        knowledge_base=PDFUrlKnowledgeBase(
            vector_db=PgVector(
                db_url=DB_URL,  
                collection="auto_rag_docs",  
                embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536, api_key=api_key),  
            ),
            num_documents=3,  
        ),
        tools=[DuckDuckGoTools()],  # Additional tool for web search via DuckDuckGo
        instructions=[
            "Search your knowledge base first.",  
            "If not found, search the internet.",  
            "Provide clear and concise answers.",  
        ],
        show_tool_calls=True,  
        search_knowledge=True,  
        markdown=True,  
        debug_mode=True,  
    )

# Function to add a PDF document to the knowledge base
def add_document(agent: Agent, file: BytesIO):
    """Add a PDF document to the agent's knowledge base.

    This function reads a PDF document from a file-like object and adds its contents to the specified agent's knowledge base. If the document is successfully read, the contents are loaded into the knowledge base with the option to upsert existing data.

    Args:
        agent (Agent): The agent whose knowledge base will be updated.
        file (BytesIO): A file-like object containing the PDF document to be added.

    Returns:
        None: The function does not return a value but provides feedback on whether the operation was successful."""
    reader = PDFReader()
