import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# Debug logs
debug_mode = st.sidebar.checkbox("🛠 Show debug logs")
st.session_state.debug = debug_mode

# Track indexed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

INDEX_PATH = "data/faiss_index/support_index.faiss"

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        with st.spinner("🔄 Processing new PDFs..."):
            summaries = handle_upload(new_files)

        for file in new_files:
            st.session_state.processed_files.add(file.name)
            st.sidebar.success(f"✅ {file.name} uploaded and indexed!")

        for filename, summary in summaries.items():
            with st.sidebar.expander(f"📄 Summary of {filename}"):
                st.markdown(summary)

# --- Sidebar Reset ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.session_state.processed_files = set()
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.rerun()

# --- Check if FAISS index exists ---
if not os.path.exists(INDEX_PATH):
    st.info("📄 Please upload support PDFs to activate SupportPilot.")
    st.stop()

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]

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
            with st.expander("📄 Sources used"):
                for chunk in context_chunks:
                    st.markdown(f"```text\n{chunk}\n```")

    st.session_state.messages.append({"role": "ai", "content": response})
