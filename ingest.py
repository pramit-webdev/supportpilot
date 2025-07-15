# ingest.py
import os
import faiss
import pickle
import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from utils import chunk_text
from llm import summarize_text

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def handle_upload(uploaded_files):
    existing_metadata = []
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)

    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
    else:
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())

    new_metadata = []
    new_embeddings = []

    summaries = {}

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        # Skip duplicate uploads
        if any(m["source"] == filename for m in existing_metadata):
            st.sidebar.info(f"⚠️ {filename} already indexed. Skipping.")
            continue

        # Save file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.sidebar.warning(f"⚠️ No text extracted from {filename}. Skipping.")
            continue

        # Summarize the document
        summary = summarize_text(text)
        summaries[filename] = summary

        chunks = chunk_text(text)
        embeddings = model.encode(chunks)

        new_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])
        new_embeddings.extend(embeddings)

        st.sidebar.success(f"✅ {filename} uploaded and indexed.")

    if new_embeddings:
        index.add(new_embeddings)
        faiss.write_index(index, INDEX_FILE)

        all_metadata = existing_metadata + new_metadata
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(all_metadata, f)

    return summaries

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
