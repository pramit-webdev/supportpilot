import os
import faiss
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from utils import chunk_text

# --- Paths ---
DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# --- Load model once ---
model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Ensure folders exist ---
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file page-by-page."""
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text
    if st.session_state.get("debug"):
        st.write(f"[DEBUG] Extracted {len(text)} characters from {file_path}")
    return text

def handle_upload(uploaded_files):
    """Processes uploaded PDFs: extract → chunk → embed → index."""
    all_chunks = []
    metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        # Save to disk
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if st.session_state.get("debug"):
            st.write(f"[DEBUG] Saved uploaded file → {file_path}")

        # Extract and chunk
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.warning(f"⚠️ Skipping {filename}: No extractable text.")
            continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        if st.session_state.get("debug"):
            st.write(f"[DEBUG] {filename} → {len(chunks)} chunks")

    # Validation
    if not all_chunks:
        st.error("❌ No valid text chunks found. Make sure PDFs contain selectable text.")
        return

    # Embed + Index
    embeddings = model.encode(all_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save to disk
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    if st.session_state.get("debug"):
        st.write(f"[DEBUG] FAISS index built with {len(all_chunks)} vectors.")
        st.write("[DEBUG] Metadata saved.")

def reset_index():
    """Deletes all uploaded documents and vector index."""
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    if st.session_state.get("debug"):
        st.write("[DEBUG] All files cleared from DOCS and INDEX directories.")
