import os
import faiss
import pickle
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from utils import chunk_text

# Paths
DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Ensure folders exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
                except Exception as e:
                    if st.session_state.get("debug"):
                        st.warning(f"[WARNING] Failed to extract from a page in {file_path}: {e}")
    except Exception as e:
        if st.session_state.get("debug"):
            st.error(f"[ERROR] Failed to open PDF {file_path}: {e}")
    return text

def handle_upload(uploaded_files):
    # Load existing index and metadata
    existing_metadata = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    indexed_files = {entry["source"] for entry in existing_metadata}

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = None

    new_chunks = []
    new_metadata = []

    for file in uploaded_files:
        filename = file.name
        if filename in indexed_files:
            st.sidebar.info(f"📁 '{filename}' already processed. Skipping.")
            continue

        # Save file to disk
        file_path = os.path.join(DOCS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if st.session_state.get("debug"):
            st.write(f"[DEBUG] Saved file: {file_path}")

        # Extract text
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.sidebar.warning(f"⚠️ No text found in '{filename}'. Skipping.")
            continue

        # Chunk text
        chunks = chunk_text(text)
        if st.session_state.get("debug"):
            st.write(f"[DEBUG] Extracted {len(chunks)} chunks from '{filename}'")

        # Embed
        embeddings = model.encode(chunks, batch_size=32, show_progress_bar=st.session_state.get("debug", False))

        # Build or update FAISS index
        if index is None:
            dim = embeddings[0].shape[0]
            index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # Store metadata
        new_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        st.sidebar.success(f"✅ '{filename}' indexed successfully!")

    # Save updated index and metadata
    if new_metadata:
        existing_metadata.extend(new_metadata)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(existing_metadata, f)
        faiss.write_index(index, INDEX_FILE)
        if st.session_state.get("debug"):
            st.write(f"[DEBUG] Saved FAISS index and metadata for {len(new_metadata)} new chunks.")

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    if st.session_state.get("debug"):
        st.write("[DEBUG] All documents and index files have been deleted.")
