import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llm import stream_llm_response

# Paths
INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def load_faiss_and_metadata():
    if not os.path.exists(INDEX_FILE) or not os.path.exists(METADATA_FILE):
        print("[ERROR] FAISS index or metadata missing.")
        return None, None
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    print(f"[DEBUG] Loaded index and metadata: {len(metadata)} entries")
    return index, metadata

def retrieve_context(query, k=3):
    index, metadata = load_faiss_and_metadata()
    if index is None:
        return "⚠️ No documents found. Please upload support PDFs first."

    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, k)

    contexts = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            chunk = metadata[idx].get("text", "")
            source = metadata[idx].get("source", "unknown")
            chunk_text = f"{chunk}\n\n(Source: {source})"
            contexts.append(chunk_text)

    print(f"[DEBUG] Retrieved {len(contexts)} relevant chunks.")
    return contexts

def get_answer(query):
    context_chunks = retrieve_context(query)
    if isinstance(context_chunks, str):  # Error string
        yield context_chunks
        return

    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI customer support assistant.

Use the following document context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    print(f"[DEBUG] Prompt sent to Groq:\n{prompt[:500]}...\n")

    yield from stream_llm_response(prompt)
