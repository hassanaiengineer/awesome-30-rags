import os

import streamlit as st
from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIChat
from agno.vectordb.lancedb import LanceDb, SearchType

from dotenv import load_dotenv

from branding import apply_streamlit_branding

# Load environment variables
load_dotenv()

apply_streamlit_branding(
    st,
    title="Agentic RAG with GPT-5",
    page_icon="🧠",
    caption="Hassan Khan RAG Series • Agentic retrieval + answering with LanceDB",
)

st.markdown("""
This app demonstrates an intelligent AI agent that:
1. **Retrieves** relevant information from knowledge sources using LanceDB
2. **Answers** your questions clearly and concisely

Enter your OpenAI API key in the sidebar to get started!
""")

# Sidebar for API key and settings
with st.sidebar:
    st.header("🔧 Configuration")
    
    # OpenAI API Key
    openai_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Used to generate embeddings + answers.",
    )

    # Add URLs to knowledge base
    st.subheader("🌐 Add Knowledge Sources")
    new_url = st.text_input(
        "Add URL",
        placeholder="https://example.com/document.pdf",
        help="Optional: add a URL to index into the knowledge base",
    )
    
    if st.button("➕ Add URL", type="primary"):
        if new_url:
            st.session_state.urls_to_add = new_url
            st.success(f"URL added to queue: {new_url}")
        else:
            st.error("Please enter a URL")

# Check if API key is provided
if openai_key:
    # Initialize URLs in session state
    if 'knowledge_urls' not in st.session_state:
        st.session_state.knowledge_urls = []
    if 'urls_loaded' not in st.session_state:
        st.session_state.urls_loaded = set()

    # Initialize knowledge base (cached to avoid reloading)
    @st.cache_resource(show_spinner="📚 Loading knowledge base...")
    def load_knowledge() -> Knowledge:
        """Load and initialize the knowledge base with LanceDB"""
        kb = Knowledge(
            vector_db=LanceDb(
                uri="tmp/lancedb",
                table_name="agentic_rag_docs",
                search_type=SearchType.vector,  # Use vector search
                embedder=OpenAIEmbedder(
                    api_key=openai_key
                ),
            ),
        )
        return kb

    # Initialize agent (cached to avoid reloading)
    @st.cache_resource(show_spinner="🤖 Loading agent...")
    def load_agent(_kb: Knowledge) -> Agent:
        """Create an agent with reasoning capabilities"""
        return Agent(
            model=OpenAIChat(
                id="gpt-5",
                api_key=openai_key
            ),
            knowledge=_kb,
            search_knowledge=True,  # Enable knowledge search
            instructions=[
                "Always search your knowledge before answering the question.",
                "Provide clear, well-structured answers in markdown format.",
                "Use proper markdown formatting with headers, lists, and emphasis where appropriate.",
                "Structure your response with clear sections and bullet points when helpful.",
            ],
            markdown=True,  # Enable markdown formatting
        )

    # Load knowledge and agent
    knowledge = load_knowledge()
    
    # Load initial URLs if any (only load once per URL)
    for url in st.session_state.knowledge_urls:
        if url not in st.session_state.urls_loaded:
            knowledge.add_content(url=url)
            st.session_state.urls_loaded.add(url)
    
    agent = load_agent(knowledge)
    
    # Display current URLs in knowledge base
    if st.session_state.knowledge_urls:
        st.sidebar.subheader("📚 Current Knowledge Sources")
        for i, url in enumerate(st.session_state.knowledge_urls, 1):
            st.sidebar.markdown(f"{i}. {url}")
    else:
        st.sidebar.info("No sources indexed yet. Add a URL to start.")
    
    # Handle URL additions
    if hasattr(st.session_state, 'urls_to_add') and st.session_state.urls_to_add:
        new_url = st.session_state.urls_to_add
        if new_url not in st.session_state.knowledge_urls:
            st.session_state.knowledge_urls.append(new_url)
        with st.spinner("📥 Loading new documents..."):
            if new_url not in st.session_state.urls_loaded:
                knowledge.add_content(url=new_url)
                st.session_state.urls_loaded.add(new_url)
        st.success(f"✅ Added: {new_url}")
        del st.session_state.urls_to_add
        st.rerun()

    # Main query section
    st.divider()
    st.subheader("🤔 Ask a Question")
    
    # Suggested prompts
    st.markdown("**Try these prompts:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Summarize sources", use_container_width=True):
            st.session_state.query = "Summarize the key points from my indexed sources."
    with col2:
        if st.button("Extract action items", use_container_width=True):
            st.session_state.query = "Extract actionable recommendations from my sources."
    with col3:
        if st.button("Find definitions", use_container_width=True):
            st.session_state.query = "List important terms and define them with supporting context."
    
    # Query input
    query = st.text_area(
        "Your question:",
        value=st.session_state.get("query", ""),
        height=100,
        help="Ask anything about the loaded knowledge sources"
    )
    
    # Run button
    if st.button("🚀 Get Answer", type="primary"):
        if query:
            # Create container for answer
            st.markdown("### 💡 Answer")
            answer_container = st.container()
            answer_placeholder = answer_container.empty()
            
            # Variables to accumulate content
            answer_text = ""
            
            # Stream the agent's response
            with st.spinner("🔍 Searching and generating answer..."):
                for chunk in agent.run(
                    query,
                    stream=True,  # Enable streaming
                ):
                    # Update answer display - show content from streaming chunks
                    if hasattr(chunk, 'content') and chunk.content and isinstance(chunk.content, str):
                        answer_text += chunk.content
                        answer_placeholder.markdown(
                            answer_text, 
                            unsafe_allow_html=True
                        )
        else:
            st.error("Please enter a question")

else:
    # Show instructions if API key is missing
    st.info("""
    👋 **Welcome! To use this app, you need:**
    
    - **OpenAI API Key** (set it in the sidebar)
    
    Once you enter the key, the app will load the knowledge base and agent.
    """)

# Footer with explanation
st.divider()
with st.expander("📖 How This Works"):
    st.markdown("""
    **This app uses the Agno framework to create an intelligent Q&A system:**
    
    1. **Knowledge Loading**: URLs are processed and stored in LanceDB vector database
    2. **Vector Search**: Uses OpenAI's embeddings for semantic search to find relevant information
    3. **GPT-5**: OpenAI's GPT-5 model processes the information and generates answers
    
    **Key Components:**
    - `Knowledge`: Manages document loading from URLs
    - `LanceDb`: Vector database for efficient similarity search
    - `OpenAIEmbedder`: Converts text to embeddings using OpenAI's embedding model
    - `Agent`: Orchestrates everything to answer questions
    
    **Why LanceDB?**
    - Lightweight and easy to set up
    - No external database required
    - Fast vector search capabilities
    - Perfect for prototyping and small to medium-scale applications
    """)
