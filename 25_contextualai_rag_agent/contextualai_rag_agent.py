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
