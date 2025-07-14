import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from llm import stream_llm_response

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
            chunk = metadata[idx].get("text", "")
            source = metadata[idx].get("source", "unknown")
            chunk_text = f"{chunk}\n\n(Source: {source})"
            contexts.append(chunk_text)

    return contexts

# Build RAG prompt and stream from Groq
def get_answer(query):
    context_chunks = retrieve_context(query)
    if isinstance(context_chunks, str):  # error message
        yield context_chunks
        return

    # Build prompt
    context = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful AI customer support assistant.

Use the following document context to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

    # Debug print (optional)
    print(f"\n[DEBUG] Prompt sent to Groq:\n{prompt[:1000]}...\n")

    # Stream from Groq
    yield from stream_llm_response(prompt)
