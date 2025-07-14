import os
import pickle
import faiss
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from utils import chunk_text

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

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
    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        raw_text = extract_text_from_pdf(file_path)
        chunks = chunk_text(raw_text)

        if st.session_state.get("debug"):
            st.write(f"[DEBUG] {filename} → {len(chunks)} chunks")

        all_chunks.extend(chunks)
        metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

    if not all_chunks:
        st.error("❌ No text extracted from uploaded files.")
        return

    embeddings = model.encode(all_chunks)

    if index is None:
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def reset_index():
    for path in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(path):
            os.remove(os.path.join(path, file))
