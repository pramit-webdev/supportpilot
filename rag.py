import os
import faiss
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from llm import call_llm

# Paths
INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
def load_faiss_and_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

# Retrieve top-k relevant context chunks
def retrieve_context(query, k=3):
    index, metadata = load_faiss_and_metadata()
    if index is None:
        return "⚠️ No documents found. Please upload support PDFs first."

    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)

    contexts = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            source = metadata[idx].get("source", "unknown")
            chunk_text = metadata[idx].get("text", "")
            context_block = f"From `{source}` →\n{chunk_text}"
            contexts.append(context_block)

    if st.session_state.get("debug"):
        st.write(f"[DEBUG] Retrieved {len(contexts)} relevant chunks.")

    return contexts

# Main RAG logic
def get_answer(query):
    context_chunks = retrieve_context(query)
    if isinstance(context_chunks, str):  # error message
        return context_chunks

    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI customer support assistant.

Use the following document context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    if st.session_state.get("debug"):
        st.write("[DEBUG] Prompt sent to LLM:")
        st.code(prompt[:1000])

    return call_llm(prompt)
