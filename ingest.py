import os
import faiss
import pickle
import pdfplumber
from sentence_transformers import SentenceTransformer
from utils import chunk_text
import streamlit as st

# Paths
DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure folders
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def handle_upload(uploaded_files):
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = None
        metadata = []

    all_chunks = []
    new_metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        if st.session_state.get("debug"):
            st.write(f"[DEBUG] Saved: {filename}")

        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        if not chunks:
            st.warning(f"⚠️ No valid text found in {filename}. Skipping.")
            continue

        embeddings = model.encode(chunks)

        if index is None:
            dim = embeddings[0].shape[0]
            index = faiss.IndexFlatL2(dim)

        index.add(embeddings)
        new_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

    metadata.extend(new_metadata)

    # Save updated index & metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    if st.session_state.get("debug"):
        st.write(f"[DEBUG] Total documents indexed: {len(metadata)}")

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
