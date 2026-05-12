# Import necessary libraries
import os

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.qdrant import Qdrant
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.os import AgentOS

# Define the collection name for the vector database
collection_name = os.getenv("QDRANT_COLLECTION", "local-rag-index").strip() or "local-rag-index"
qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333/").strip() or "http://localhost:6333/"

# Set up Qdrant as the vector database with the embedder
vector_db = Qdrant(
    collection=collection_name,
    url=qdrant_url,
    embedder=OllamaEmbedder()
)

# Define the knowledge base
knowledge_base = Knowledge(
    vector_db=vector_db,
)

# Optional: preload a single source URL via env var (no default baked in).
source_url = os.getenv("RAG_SOURCE_URL", "").strip()
if source_url:
    knowledge_base.add_content(url=source_url)

# Create the Agent using Ollama's llama3.2 model and the knowledge base
agent = Agent(
    name="Hassan Khan • Local RAG Agent",
    model=Ollama(id="llama3.2"),
    knowledge=knowledge_base,
)

# UI for RAG agent
agent_os = AgentOS(agents=[agent])
app = agent_os.get_app()

# Run the AgentOS app
if __name__ == "__main__":
    agent_os.serve(app="local_rag_agent:app", reload=True)
