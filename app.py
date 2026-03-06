"""
app.py
HOA RAG Chatbot - Streamlit Chat Interface

A simple chat interface for HOA members to ask questions about
policies, bylaws, and guidelines.

Usage: streamlit run app.py
"""

import streamlit as st
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from query import load_collection, answer_question

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Duck Bot (UCI Assistant)",
    page_icon="🏘️",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .stChatMessage { border-radius: 10px; }
        .block-container { max-width: 800px; }
        footer { visibility: hidden; }
            /* Targets the sidebar container */
        [data-testid="stSidebar"] {
            background-color: #545454; /* Example: Red background */
            color: white; /* Example: White text color */
        }
            [alt=Logo] {
            height: 3rem; /* Adjust this value (e.g., 2rem, 3rem, 100px) */
        }
    </style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("duckBot. :duck:")
st.caption("Ask questions about UCI policies, bylaws, and guidelines. ")
st.caption("Answers are based on official Ulmstead Club documents.")
st.divider()
st.logo('images/logo.png')

# ── Load ChromaDB (cached so it only loads once) ──────────────────────────────
@st.cache_resource
def get_collection():
    try:
        return load_collection()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}")
        st.stop()

collection = get_collection()

# ── Session State for Chat History ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi! I can answer questions about Ulmstead Club policies, "
                "bylaws, and guidelines. What would you like to know?"
            )
        }
    ]

# ── Render Chat History ───────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
if question := st.chat_input("Ask a question about UCI policies..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer = answer_question(question, collection)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.markdown("""
    Examples
    * Can a renter vote at HOA meetings?
    * What happens if I build a deck without approval?
    * "Who is eligible to select a boat slip?"
    """)
    st.markdown("""
    This assistant searches the following official Ulmstead Club documents:

    - 📄 Bylaws
    - 📄 Covenants
    - 📄 Architecture Guidelines
    - ⛵ Nautical Guidelines
    - 🏖️ Beach Park Guidelines
    - 👤 Membership Policy

    Answers include citations to the relevant article or section where possible.
    """)

    st.divider()
    st.markdown(
        "For official decisions or disputes, please contact the HOA board directly."
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Hi! I can answer questions about Ulmstead Club policies, "
                    "bylaws, and guidelines. What would you like to know?"
                )
            }
        ]
        st.rerun()