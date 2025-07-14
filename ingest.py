import os
import faiss
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from utils import chunk_text

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
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
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

    if not all_chunks:
        st.error("❌ No valid text found in PDFs.")
        return

    embeddings = model.encode(all_chunks)
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
