import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llm import call_llm

INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_index_and_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve_context(query, k=3):
    index, metadata = load_index_and_metadata()
    if index is None or metadata is None:
        return "⚠️ No index found. Please upload documents.", []
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)
    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            source = metadata[idx]["source"]
            text = metadata[idx]["text"]
            context_chunks.append(f"From **{source}**:\n{text.strip()[:500]}")
    return context_chunks

def get_answer(query):
    context_chunks = retrieve_context(query)
    if isinstance(context_chunks, str):
        return context_chunks, []
    context_str = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI assistant.

Use the following context from documents to answer the user's question.

Context:

{context_str}

Question: {query}

Answer:"""
    answer = call_llm(prompt)
    return answer, context_chunks
