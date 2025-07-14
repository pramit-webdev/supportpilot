import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

st.set_page_config(page_title="SupportPilot", layout="wide")
st.title("🤖 SupportPilot – AI Customer Support Assistant")

# --- Paths ---
INDEX_PATH = "data/faiss_index/support_index.faiss"

# --- Sidebar Upload ---
st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    handle_upload(uploaded_files)
    st.sidebar.success("✅ Documents uploaded and indexed.")

# --- Sidebar Reset ---
if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("🧹 All documents and index cleared.")

# --- Check if FAISS index exists ---
index_ready = os.path.exists(INDEX_PATH)

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm SupportPilot. Upload your PDFs and ask me anything!"}
    ]

# --- Display Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Block chat until documents are uploaded ---
if not index_ready:
    st.info("📄 Please upload support PDFs from the sidebar to activate SupportPilot.")
    st.stop()

# --- Chat Input ---
user_input = st.chat_input("Type your question...")

if user_input:
    # Save user input
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream AI reply
    with st.chat_message("ai"):
        response_stream = get_answer(user_input)
        full_response = st.write_stream(response_stream)

    # Save AI response
    st.session_state.messages.append({"role": "ai", "content": full_response})
