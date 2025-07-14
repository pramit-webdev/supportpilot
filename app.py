import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

# --- Page Settings ---
st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# --- Debug Logs Toggle ---
debug_mode = st.sidebar.checkbox("🛠 Show debug logs")
st.session_state.debug = debug_mode

# --- Paths ---
INDEX_PATH = "data/faiss_index/support_index.faiss"

# --- Session State for File Tracking ---
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            with st.spinner(f"🔄 Indexing {file.name}..."):
                handle_upload([file])  # process one by one
            st.sidebar.success(f"✅ {file.name} indexed successfully.")
            st.session_state.processed_files.add(file.name)
    st.rerun()  # Show updated chat state after new docs

# --- Sidebar: Reset Button ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.session_state.processed_files.clear()
    st.session_state.messages = []
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.rerun()

# --- Block Chat If No Index Exists ---
if not os.path.exists(INDEX_PATH):
    st.info("📄 Please upload support PDFs to activate SupportPilot.")
    st.stop()

# --- Chat History Init ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]

# --- Display Chat Messages (ChatGPT-style) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input ---
user_input = st.chat_input("Type your question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        with st.spinner("💬 Thinking..."):
            response, context_sources = get_answer(user_input)
            st.markdown(response)

            if context_sources:
                with st.expander("📄 Sources used"):
                    for src in context_sources:
                        st.markdown(f"- `{src}`")

    st.session_state.messages.append({"role": "ai", "content": response})
