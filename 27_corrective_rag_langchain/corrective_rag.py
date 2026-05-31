from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from pydantic import BaseModel, Field
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_community.tools import TavilySearchResults
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage        
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict
from langchain_core.prompts import PromptTemplate
import pprint
import yaml
import nest_asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import tempfile
import os
from langchain_anthropic import ChatAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from branding import apply_streamlit_branding


nest_asyncio.apply()

retriever = None

apply_streamlit_branding(
    st,
    title="Corrective RAG (LangGraph)",
    page_icon="🧯",
    caption="Hassan Khan RAG Series • Retrieval + self-correction + optional web search",
)

def initialize_session_state():
    """Initialize session state variables for API keys and URLs."""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        # Initialize API keys and URLs
        st.session_state.anthropic_api_key = ""
        st.session_state.openai_api_key = ""
        st.session_state.tavily_api_key = ""
        st.session_state.qdrant_api_key = ""
        st.session_state.qdrant_url = "http://localhost:6333"
        st.session_state.doc_url = ""
        
def setup_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        st.session_state.anthropic_api_key = st.text_input("Anthropic API Key", value=st.session_state.anthropic_api_key, type="password", help="Required for Claude 3 model")
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        st.session_state.tavily_api_key = st.text_input("Tavily API Key", value=st.session_state.tavily_api_key, type="password")
        st.session_state.qdrant_url = st.text_input("Qdrant URL", value=st.session_state.qdrant_url)
        st.session_state.qdrant_api_key = st.text_input("Qdrant API Key", value=st.session_state.qdrant_api_key, type="password")
        st.session_state.doc_url = st.text_input("Document URL", value=st.session_state.doc_url)
        
        if not all([st.session_state.openai_api_key, st.session_state.anthropic_api_key, st.session_state.qdrant_url]):
            st.warning("Please provide the required API keys and URLs")
            st.stop()
        
        st.session_state.initialized = True

initialize_session_state()
setup_sidebar()

# Use session state variables instead of config
openai_api_key = st.session_state.openai_api_key
tavily_api_key = st.session_state.tavily_api_key
anthropic_api_key = st.session_state.anthropic_api_key

# Update embeddings initialization
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=st.session_state.openai_api_key
)

# Update Qdrant client initialization
client = QdrantClient(
    url=st.session_state.qdrant_url,
    api_key=st.session_state.qdrant_api_key
)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def execute_tavily_search(tool, query):
    return tool.invoke({"query": query})

def web_search(state):
    """Web search based on the re-phrased question using Tavily API."""
    print("~-web search-~")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    progress_placeholder.info("Initiating web search...")
    
    try:
        # Validate Tavily API key
        if not st.session_state.tavily_api_key:
            progress_placeholder.warning("Tavily API key not provided - skipping web search")
            return {"keys": {"documents": documents, "question": question}}
        
        progress_placeholder.info("Configuring search tool...")
        
        # Initialize Tavily search tool
        tool = TavilySearchResults(
            api_key=st.session_state.tavily_api_key,
            max_results=3,
            search_depth="advanced"
        )
        
        # Execute search with retry logic
        progress_placeholder.info("Executing search query...")
        try:
            search_results = execute_tavily_search(tool, question)
        except Exception as search_error:
            progress_placeholder.error(f"Search failed after retries: {str(search_error)}")
            return {"keys": {"documents": documents, "question": question}}
        
        if not search_results:
            progress_placeholder.warning("No search results found")
            return {"keys": {"documents": documents, "question": question}}
        
        # Process results
        progress_placeholder.info("Processing search results...")
        web_results = []
        for result in search_results:
            # Extract and format relevant information
            content = (
                f"Title: {result.get('title', 'No title')}\n"
                f"Content: {result.get('content', 'No content')}\n"
            )
            web_results.append(content)
        
        # Create document from results
        web_document = Document(
            page_content="\n\n".join(web_results),
            metadata={
                "source": "tavily_search",
                "query": question,
                "result_count": len(web_results)
            }
        )
        documents.append(web_document)
        
        progress_placeholder.success(f"Successfully added {len(web_results)} search results")
        
