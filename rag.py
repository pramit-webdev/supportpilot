import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llm import call_llm

INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"

model = SentenceTransformer("all-mpnet-base-v2")

def load_index_and_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, []
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve_context(query, k=10):
    index, metadata = load_index_and_metadata()
    if index is None or not metadata:
        return [], []
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            chunk = f"[{entry['source']}]: {entry['text'].strip()}"
            context_chunks.append(chunk)
    return context_chunks

def get_answer(query):
    context_chunks = retrieve_context(query)
    if not context_chunks:
        return "⚠️ No relevant context found. Please try a different question.", []
    context_str = "\n---\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. Use the context below to answer accurately.\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    answer = call_llm(prompt)
    return answer, context_chunks
