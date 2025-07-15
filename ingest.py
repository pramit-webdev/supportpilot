import os
import faiss
import pickle
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from utils import chunk_text, generate_summary
import numpy as np

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        if st.session_state.get("debug"):
            st.write(f"[ERROR] Failed to read {file_path}: {e}")
    return text

def handle_upload(uploaded_files):
    all_chunks = []
    all_embeddings = []
    all_metadata = []
    summaries = {}

    # Load existing index and metadata
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        index = None
        existing_metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if st.session_state.get("debug"):
            st.write(f"[DEBUG] Saved {filename}")

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.sidebar.warning(f"⚠️ Skipped {filename}: No text found.")
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks, show_progress_bar=False)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        # Store summary
        summaries[filename] = generate_summary(text)

    # Skip if no valid docs
    if not all_embeddings:
        st.sidebar.error("❌ No valid documents to index.")
        return {}

    # Merge into index
    if index is None:
        dim = all_embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dim)

    embeddings_np = np.array(all_embeddings).astype("float32")
    index.add(embeddings_np)

    faiss.write_index(index, INDEX_FILE)

    combined_metadata = existing_metadata + all_metadata
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(combined_metadata, f)

    return summaries

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    if st.session_state.get("debug"):
        st.write("[DEBUG] Reset completed.")
