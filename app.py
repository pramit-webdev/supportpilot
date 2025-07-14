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

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("🔄 Indexing documents..."):
        handle_upload(uploaded_files)
    st.sidebar.success("✅ Documents uploaded and indexed.")

# --- Sidebar: Reset Button ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.rerun()  # Clear chat + UI

# --- Check if FAISS index exists ---
index_ready = os.path.exists(INDEX_PATH)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]

# --- Display Chat Messages (ChatGPT style) ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Block Chat Input If No PDFs Uploaded ---
if not index_ready:
    st.info("📄 Please upload support PDFs from the sidebar to activate SupportPilot.")
    st.stop()

# --- Chat Input ---
user_input = st.chat_input("Type your question...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get full AI response (non-streaming)
    with st.chat_message("ai"):
        full_response = get_answer(user_input)
        st.markdown(full_response)

    # Add AI response to chat history
    st.session_state.messages.append({"role": "ai", "content": full_response})
