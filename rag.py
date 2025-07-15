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
        return "⚠️ No documents found. Please upload PDFs first.", []

    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)

    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            source = metadata[idx]["source"]
            text = metadata[idx]["text"]
            context_chunks.append(f"From **{source}**:\n{text.strip()[:500]}")
    return None, context_chunks

def get_answer(query):
    error_msg, context_chunks = retrieve_context(query)
    if error_msg:
        return error_msg, []

    context_text = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI customer support assistant.

Use the following document context to answer the user's question.

Context:
{context_text}

Question: {query}
Answer:"""

    answer = call_llm(prompt)
    return answer, context_chunks
