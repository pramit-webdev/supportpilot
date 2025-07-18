import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
import os

st.set_page_config(page_title="DocuPilot", layout="wide")
st.title("🤖 DocuPilot – Your AI Document Assistant")

INDEX_PATH = "data/faiss_index/support_index.faiss"

with st.sidebar:
    st.header("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, Word Docs, CSVs, or Excel",
        type=["pdf", "docx", "doc", "csv", "xlsx"],
        accept_multiple_files=True
    )
    if st.button("🗑️ Reset all documents"):
        reset_index()
        st.warning("Storage/Index cleared. Upload fresh documents.")
        st.rerun()

if uploaded_files:
    with st.spinner("Processing and indexing files..."):
        summaries = handle_upload(uploaded_files)
    for fname, summary in summaries.items():
        st.sidebar.success(f"Indexed: {fname}")
        with st.sidebar.expander(f"Summary of {fname}"):
            st.markdown(summary)

if not os.path.exists(INDEX_PATH):
    st.info("Upload documents to activate DocuPilot.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! Upload your PDFs, Word docs, CSVs, or Excel files and ask a question!"}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_input = st.chat_input("Ask a question...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("ai"):
        with st.spinner("Searching and generating answer..."):
            answer, context_chunks = get_answer(user_input)
            st.markdown(answer)
            with st.expander("Context Used"):
                for i, chunk in enumerate(context_chunks, 1):
                    st.markdown(f"**Chunk {i}:**  {chunk}")
    st.session_state.messages.append({"role": "ai", "content": answer})
