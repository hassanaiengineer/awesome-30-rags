import os
import tempfile
import time
from typing import List, Optional, Tuple, Any

import streamlit as st
import requests
import json
import re
from contextual import ContextualAI

from branding import apply_streamlit_branding

apply_streamlit_branding(
    st,
    title="ContextualAI RAG Agent",
    page_icon="🧠",
    caption="Hassan Khan RAG Series • Managed agent + datastore workflow",
)


def init_session_state() -> None:
    if "api_key_submitted" not in st.session_state:
        st.session_state.api_key_submitted = False
    if "contextual_api_key" not in st.session_state:
        st.session_state.contextual_api_key = ""
    if "base_url" not in st.session_state:
        st.session_state.base_url = "https://api.contextual.ai/v1"
    if "agent_id" not in st.session_state:
        st.session_state.agent_id = ""
    if "datastore_id" not in st.session_state:
        st.session_state.datastore_id = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_file" not in st.session_state:
        st.session_state.processed_file = False
    if "last_raw_response" not in st.session_state:
        st.session_state.last_raw_response = None
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = ""


def sidebar_api_form() -> bool:
    with st.sidebar:
        st.header("API & Resource Setup")

        if st.session_state.api_key_submitted:
            st.success("API verified")
            if st.button("Reset Setup"):
                st.session_state.clear()
                st.rerun()
            return True

        with st.form("contextual_api_form"):
            api_key = st.text_input("Contextual AI API Key", type="password")
            base_url = st.text_input(
                "Base URL",
                value=st.session_state.base_url,
                help="Include /v1 (e.g., https://api.contextual.ai/v1)",
            )
            existing_agent_id = st.text_input("Existing Agent ID (optional)")
            existing_datastore_id = st.text_input("Existing Datastore ID (optional)")

            if st.form_submit_button("Save & Verify"):
                try:
                    client = ContextualAI(api_key=api_key, base_url=base_url)
                    _ = client.agents.list()

                    st.session_state.contextual_api_key = api_key
                    st.session_state.base_url = base_url
                    st.session_state.agent_id = existing_agent_id
                    st.session_state.datastore_id = existing_datastore_id
                    st.session_state.api_key_submitted = True

                    st.success("Credentials verified!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Credential verification failed: {str(e)}")
        return False


def ensure_client():
    if not st.session_state.get("contextual_api_key"):
        raise ValueError("Contextual AI API key not provided")
    return ContextualAI(api_key=st.session_state.contextual_api_key, base_url=st.session_state.base_url)


def create_datastore(client, name: str) -> Optional[str]:
    try:
        ds = client.datastores.create(name=name)
        return getattr(ds, "id", None)
    except Exception as e:
        st.error(f"Failed to create datastore: {e}")
        return None


ALLOWED_EXTS = {".pdf", ".html", ".htm", ".mhtml", ".doc", ".docx", ".ppt", ".pptx"}

def upload_documents(client, datastore_id: str, files: List[bytes], filenames: List[str], metadata: Optional[dict]) -> List[str]:
    doc_ids: List[str] = []
    for content, fname in zip(files, filenames):
        try:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ALLOWED_EXTS:
                st.error(f"Unsupported file extension for {fname}. Allowed: {sorted(ALLOWED_EXTS)}")
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            with open(tmp_path, "rb") as f:
