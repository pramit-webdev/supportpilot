# ✅ app.py (with filters and follow-up memory)

import os
import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer

st.set_page_config(page_title="DocuPilot", layout="wide")
st.title("🤖 DocuPilot – Your AI Document Assistant")

INDEX_PATH = "data/faiss_index/support_index.faiss"

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

with st.sidebar:
    st.header("📁 Upload Your Files")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, CSV, XLSX",
        type=["pdf", "docx", "doc", "csv", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("🗑️ Reset All Documents"):
        reset_index()
        st.session_state.indexed_files = set()
        st.warning("All files and indexes have been cleared.")
        st.rerun()

    if st.session_state.indexed_files:
        st.sidebar.markdown("### 🔍 Filter Search Scope")

        selected_files = st.sidebar.multiselect(
            "Search only in these files:",
            options=list(st.session_state.indexed_files),
            default=list(st.session_state.indexed_files)
        )

        selected_types = st.sidebar.multiselect(
            "File types:",
            options=[".pdf", ".docx", ".csv", ".xlsx"],
            default=[]
        )
    else:
        selected_files = []
        selected_types = []

# Index only new files
new_files = []
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.indexed_files:
            new_files.append(file)

    if new_files:
        with st.spinner("Indexing new files..."):
            summaries = handle_upload(new_files)
        for file in new_files:
            st.session_state.indexed_files.add(file.name)
        for fname, summary in summaries.items():
            st.sidebar.success(f"Indexed: {fname}")
            with st.sidebar.expander(f"Summary - {fname}"):
                st.markdown(summary)

if not os.path.exists(INDEX_PATH):
    st.info("Upload documents to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! Upload files and ask a question about their contents."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        with st.spinner("Finding answer..."):
            answer, context = get_answer(user_input, selected_files, selected_types)
            st.markdown(answer)
            with st.expander("📄 Context Chunks"):
                for i, chunk in enumerate(context, 1):
                    st.markdown(f"**Chunk {i}**\n\n{chunk}")
    st.session_state.messages.append({"role": "ai", "content": answer})
