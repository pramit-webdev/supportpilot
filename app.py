import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# Debug logs
debug_mode = st.sidebar.checkbox("🛠 Show debug logs")
st.session_state.debug = debug_mode

# Track whether indexing is complete
if "files_indexed" not in st.session_state:
    st.session_state.files_indexed = False

# FAISS index file path
INDEX_PATH = "data/faiss_index/support_index.faiss"

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and not st.session_state.files_indexed:
    with st.spinner("🔄 Processing uploaded PDFs..."):
        handle_upload(uploaded_files)
    st.sidebar.success("✅ All files indexed successfully.")
    st.session_state.files_indexed = True
    st.rerun()

# --- Sidebar Reset ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.session_state.files_indexed = False
    st.rerun()

# --- Index must exist to chat ---
index_ready = os.path.exists(INDEX_PATH)
if not index_ready:
    st.info("📄 Please upload support PDFs to activate SupportPilot.")
    st.stop()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]

# --- Display Past Messages ---
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
        with st.spinner("💬 Generating response..."):
            response, context_chunks = get_answer(user_input)
            st.markdown(response)

            # Collapsible source viewer
            with st.expander("📄 Sources used"):
                for chunk in context_chunks:
                    st.markdown(f"```text\n{chunk}\n```")

    st.session_state.messages.append({"role": "ai", "content": response})
