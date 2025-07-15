import os
import faiss
import pickle
import pdfplumber
import numpy as np
from sentence_transformers import SentenceTransformer
from utils import chunk_text
from llm import call_llm
import streamlit as st

DOCS_DIR = "data/docs"
INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs("data/faiss_index", exist_ok=True)

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def generate_summary(text):
    prompt = f"Summarize the following document in one paragraph:\n\n{text[:3000]}"
    return call_llm(prompt)

def handle_upload(uploaded_files):
    summaries = {}
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        index = None
        existing_metadata = []

    new_chunks = []
    new_metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.sidebar.warning(f"⚠️ No extractable text in {filename}. Skipped.")
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks, convert_to_numpy=True)
        new_chunks.extend(embeddings)
        new_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])
        summary = generate_summary(text)
        summaries[filename] = summary
        st.sidebar.success(f"✅ {filename} uploaded and indexed.")

    if not new_chunks:
        return summaries

    new_embeddings = np.array(new_chunks).astype("float32")

    if index is None:
        index = faiss.IndexFlatL2(new_embeddings.shape[1])
    index.add(new_embeddings)

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(existing_metadata + new_metadata, f)

    return summaries

def reset_index():
    for folder in ["data/docs", "data/faiss_index"]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
