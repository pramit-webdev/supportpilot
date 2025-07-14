import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

# --- Page Settings ---
st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# --- Debug Toggle ---
debug_mode = st.sidebar.checkbox("🛠 Show debug logs")
st.session_state.debug = debug_mode

# --- Paths ---
INDEX_PATH = "data/faiss_index/support_index.faiss"

# --- Init Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]
if "files_indexed" not in st.session_state:
    st.session_state.files_indexed = False

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files and not st.session_state.files_indexed:
    with st.spinner("🔄 Indexing documents..."):
        handle_upload(uploaded_files)
    st.sidebar.success("✅ Documents uploaded and indexed.")
    st.session_state.files_indexed = True

# --- Sidebar: Reset Button ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.session_state.files_indexed = False
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.rerun()

# --- Check FAISS Index Exists ---
index_ready = os.path.exists(INDEX_PATH)

# --- Display Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Block Chat if No Index ---
if not index_ready:
    st.info("📄 Please upload support PDFs from the sidebar to activate SupportPilot.")
    st.stop()

# --- Chat Input ---
user_input = st.chat_input("Type your question...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get AI response
    with st.chat_message("ai"):
        with st.spinner("💬 SupportPilot is thinking..."):
            full_response, source_chunks = get_answer(user_input)
        st.markdown(full_response)

        # Show context viewer
        with st.expander("🔍 Show sources used to answer this question"):
            for chunk in source_chunks:
                st.markdown(f"```text\n{chunk}\n```")

    # Add to chat history
    st.session_state.messages.append({"role": "ai", "content": full_response})
