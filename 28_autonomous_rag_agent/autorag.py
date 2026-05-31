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

