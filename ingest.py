import os
import faiss
import pickle
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from utils import chunk_text

# Paths
DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def handle_upload(uploaded_files):
    all_chunks = []
    metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        # Save file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Extract text
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.warning(f"⚠️ No readable text found in {filename}. Skipping.")
            continue

        # Chunk and track
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        # Success message per file
        st.success(f"✅ {filename} indexed successfully.")

    if not all_chunks:
        st.error("❌ No valid text chunks found in any PDF.")
        return

    # Encode and index
    embeddings = model.encode(all_chunks)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
