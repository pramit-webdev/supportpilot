import os
import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer

st.set_page_config(page_title="DocuPilot", layout="wide")
st.title("🤖 DocuPilot – Your AI Document Assistant")

INDEX_PATH = "data/faiss_index/support_index.faiss"

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Upload Your Files")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, CSV, XLSX",
        type=["pdf", "docx", "doc", "csv", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("🗑️ Reset All Documents"):
        reset_index()
        st.warning("All files and indexes have been cleared.")
        st.rerun()

# Index files only once
if uploaded_files and not st.session_state.get("files_indexed"):
    with st.spinner("Indexing files..."):
        summaries = handle_upload(uploaded_files)
    st.session_state.files_indexed = True
    for fname, summary in summaries.items():
        st.sidebar.success(f"Indexed: {fname}")
        with st.sidebar.expander(f"Summary - {fname}"):
            st.markdown(summary)

# Stop if no index is present
if not os.path.exists(INDEX_PATH):
    st.info("Upload documents to get started.")
    st.stop()

# Initialize chat state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! Upload files and ask a question about their contents."}
    ]

# Display previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User question input
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        with st.spinner("Finding answer..."):
            answer, context = get_answer(user_input)
            st.markdown(answer)
            with st.expander("📄 Context Chunks"):
                for i, chunk in enumerate(context, 1):
                    st.markdown(f"**Chunk {i}**\n\n{chunk}")
    st.session_state.messages.append({"role": "ai", "content": answer})
