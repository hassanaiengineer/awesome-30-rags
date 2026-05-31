from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from uuid import uuid4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from functools import partial

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from pydantic import BaseModel, Field

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition

import streamlit as st
from branding import apply_streamlit_branding

apply_streamlit_branding(
    st,
    title="AI Blog Search (LangGraph)",
    page_icon="🔎",
    caption="Hassan Khan RAG Series • Agentic RAG over web pages with LangGraph + Qdrant",
)

# Initialize session state variables if they don't exist
if 'qdrant_host' not in st.session_state:
    st.session_state.qdrant_host = ""
if 'qdrant_api_key' not in st.session_state:
    st.session_state.qdrant_api_key = ""
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

def set_sidebar():
    """Setup sidebar for API keys and configuration."""
    with st.sidebar:
        st.subheader("API Configuration")
        
        qdrant_host = st.text_input("Enter your Qdrant Host URL:", type="password")
        qdrant_api_key = st.text_input("Enter your Qdrant API key:", type="password")
        gemini_api_key = st.text_input("Enter your Gemini API key:", type="password")

        if st.button("Done"):
            if qdrant_host and qdrant_api_key and gemini_api_key:
                st.session_state.qdrant_host = qdrant_host
                st.session_state.qdrant_api_key = qdrant_api_key
                st.session_state.gemini_api_key = gemini_api_key
                st.success("API keys saved!")
            else:
                st.warning("Please fill all API fields")

def initialize_components():
    """Initialize components that require API keys"""
    if not all([st.session_state.qdrant_host, 
