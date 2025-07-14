import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llm import query_llm

# Paths
INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index and metadata
def load_faiss_and_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None

    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

def retrieve_context(query, k=3):
    index, metadata = load_faiss_and_metadata()
    if index is None:
        return "No documents have been uploaded yet. Please upload PDFs."

    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)

    contexts = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            source = metadata[idx].get("source", "unknown")
            text = f"(From `{source}`): {idx}"
            contexts.append(text)
    return contexts

def get_answer(query):
    context_chunks = retrieve_context(query)
    if isinstance(context_chunks, str):  # error string
        return context_chunks

    # Combine top chunks into a single context
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful customer support AI agent.

Use the following document context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    return query_llm(prompt)
