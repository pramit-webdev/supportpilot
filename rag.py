import os, faiss, pickle
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

def retrieve_context(query, k=12):
    index, metadata = load_index_and_metadata()
    if index is None or not metadata:
        return [], []
    query_vector = model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    context_chunks = []
    for idx in indices[0]:
        if 0 <= idx < len(metadata):
            entry = metadata[idx]
            chunk = f"From **{entry['source']}**:\n{entry['text'][:700].strip()}"
            context_chunks.append(chunk)
    print(f"🔍 Top-k ({k}) chunks for: {query}")
    for i, c in enumerate(context_chunks[:3]):
        print(f"-- Chunk {i+1}: {c[:250]}...\n")
    return context_chunks

def get_answer(query):
    context_chunks = retrieve_context(query)
    if not context_chunks:
        return "⚠️ No relevant context found. Check uploads!", []
    context_str = "\n\n".join(context_chunks)
    prompt = (
        f"You are a document QA expert. Answer the following question using ONLY the provided context. "
        f"For every answer, cite the specific source(s) and the relevant passage. "
        f"If the answer cannot be found verbatim, say 'The answer is not available in the documents.'\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )
    answer = call_llm(prompt)
    return answer, context_chunks
