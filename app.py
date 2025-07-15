import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

# --- Page Setup ---
st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()
if "debug" not in st.session_state:
    st.session_state.debug = False

# --- Debug Mode ---
st.session_state.debug = st.sidebar.checkbox("🛠 Show debug logs")

# --- Constants ---
INDEX_PATH = "data/faiss_index/support_index.faiss"

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.processed_files:
            with st.spinner(f"🔄 Indexing `{file.name}`..."):
                handle_upload([file])
            st.sidebar.success(f"✅ `{file.name}` indexed successfully.")
            st.session_state.processed_files.add(file.name)
    st.rerun()

# --- Sidebar Reset ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]
    st.session_state.processed_files.clear()
    st.rerun()

# --- Ensure at least one file is indexed ---
if not os.path.exists(INDEX_PATH):
    st.info("📄 Please upload support PDFs to activate SupportPilot.")
    st.stop()

# --- Display Chat History ---
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

            # Show context used in a collapsible section
            with st.expander("📄 Sources used"):
                for chunk in context_chunks:
                    st.markdown(f"```text\n{chunk}\n```")

    st.session_state.messages.append({"role": "ai", "content": response})
