import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

st.set_page_config(page_title="DocuPilot", layout="wide")
st.title("🤖 DocuPilot – Your AI Document Assistant")

debug_mode = st.sidebar.checkbox("🛠 Show debug logs")
st.session_state.debug = debug_mode

if "files_indexed" not in st.session_state:
    st.session_state.files_indexed = False

INDEX_PATH = "data/faiss_index/support_index.faiss"

st.sidebar.header("📁 Upload Knowledge Base")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, Word docs, CSVs, or Images",
    type=["pdf", "docx", "doc", "csv", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("🔄 Processing files..."):
        summaries = handle_upload(uploaded_files)
    for filename, summary in summaries.items():
        st.sidebar.success(f"✅ {filename} uploaded and indexed.")
        with st.sidebar.expander(f"📄 Summary of {filename}"):
            st.markdown(summary)

if st.sidebar.button("🗑️ Reset All Documents"):
    reset_index()
    st.sidebar.warning("🧹 All documents and index cleared.")
    st.rerun()

index_ready = os.path.exists(INDEX_PATH)
if not index_ready:
    st.info("📄 Please upload support PDFs, Word docs, CSVs, or images to activate DocuPilot.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! I'm DocuPilot. Upload your PDFs, Word docs, CSVs, or images, then ask me anything!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

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
                    st.markdown(chunk)
    st.session_state.messages.append({"role": "ai", "content": response})
