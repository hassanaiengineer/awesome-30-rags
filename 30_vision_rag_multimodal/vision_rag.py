import requests
import os
import io
import base64
import PIL
from PIL import Image
import tqdm
import numpy as np
import streamlit as st
import cohere
from google import genai
import fitz # PyMuPDF

from branding import apply_streamlit_branding

# --- Streamlit App Configuration ---
apply_streamlit_branding(
    st,
    title="Vision RAG with Cohere Embed-4",
    page_icon="🖼️",
    caption="Hassan Khan RAG Series • Multimodal retrieval over images/PDFs",
)

# --- API Key Input ---
with st.sidebar:
    st.header("🔑 API Keys")
    cohere_api_key = st.text_input("Cohere API Key", type="password", key="cohere_key")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password", key="google_key")

    st.markdown("---")
    if not cohere_api_key:
        st.warning("Please enter your Cohere API key to proceed.")
    if not google_api_key:
        st.warning("Please enter your Google API key to proceed.")
    st.markdown("---")


# --- Initialize API Clients ---
co = None
genai_client = None
# Initialize Session State for embeddings and paths
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = []
if 'doc_embeddings' not in st.session_state:
    st.session_state.doc_embeddings = None

if cohere_api_key and google_api_key:
    try:
        co = cohere.ClientV2(api_key=cohere_api_key)
        st.sidebar.success("Cohere Client Initialized!")
    except Exception as e:
        st.sidebar.error(f"Cohere Initialization Failed: {e}")

    try:
        genai_client = genai.Client(api_key=google_api_key)
        st.sidebar.success("Gemini Client Initialized!")
    except Exception as e:
        st.sidebar.error(f"Gemini Initialization Failed: {e}")
else:
    st.info("Enter your API keys in the sidebar to start.")

# Information about the models
with st.expander("ℹ️ About the models used"):
    st.markdown("""
    ### Cohere Embed-4
    
    Cohere's Embed-4 is a state-of-the-art multimodal embedding model designed for enterprise search and retrieval. 
    It enables:
    
    - **Multimodal search**: Search text and images together seamlessly
    - **High accuracy**: State-of-the-art performance for retrieval tasks
    - **Efficient embedding**: Process complex images like charts, graphs, and infographics
    
    The model processes images without requiring complex OCR pre-processing and maintains the connection between visual elements and text.
    
    ### Google Gemini 2.5 Flash
    
    Gemini 2.5 Flash is Google's efficient multimodal model that can process text and image inputs to generate high-quality responses.
    It's designed for fast inference while maintaining high accuracy, making it ideal for real-time applications like this RAG system.
    """)

# --- Helper functions ---
# Some helper functions to resize images and to convert them to base64 format
max_pixels = 1568*1568  #Max resolution for images

# Resize too large images
def resize_image(pil_image: PIL.Image.Image) -> None:
    """Resizes the image in-place if it exceeds max_pixels."""
    org_width, org_height = pil_image.size

    # Resize image if too large
    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

# Convert images to a base64 string before sending it to the API
def base64_from_image(img_path: str) -> str:
    """Converts an image file to a base64 encoded string."""
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else "PNG"

    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data

# Convert PIL image to base64 string
def pil_to_base64(pil_image: PIL.Image.Image) -> str:
    """Converts a PIL image to a base64 encoded string."""
    if pil_image.format is None:
        img_format = "PNG"
    else:
        img_format = pil_image.format
    
    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64,"+base64.b64encode(img_buffer.read()).decode("utf-8")

    return img_data

# Compute embedding for an image
@st.cache_data(ttl=3600, show_spinner=False)
def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    """Computes an embedding for an image using Cohere's Embed-4 model."""
    try:
        api_response = _cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[base64_img],
        )
        
        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        else:
            st.warning("Could not get embedding. API response might be empty.")
            return None
    except Exception as e:
        st.error(f"Error computing embedding: {e}")
        return None

# Process a PDF file: extract pages as images and embed them
# Note: Caching PDF processing might be complex due to potential large file sizes and streams
