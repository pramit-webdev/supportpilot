import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from utils import chunk_text, summarize_text
import faiss
import numpy as np
import pdfplumber
import pandas as pd
import docx

DATA_DIR = "data"
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

SUPPORTED_EXTS = [".pdf", ".docx", ".doc", ".csv", ".xlsx"]

model = SentenceTransformer("all-mpnet-base-v2")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_file(file_path, file_ext):
    ext = file_ext.lower()
    text = ""
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif ext in [".docx", ".doc"]:
            doc = docx.Document(file_path)
            text += "\n".join(par.text for par in doc.paragraphs)
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            text = df.to_string(index=False)
    except Exception as e:
        print(f"Text extraction error for {file_path}: {e}")
    return text

def handle_upload(uploaded_files):
    summaries = {}
    all_chunks, all_metadata = [], []

    # Load existing index/metadata
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        index, existing_metadata = None, []

    for file in uploaded_files:
        filename = file.name
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SUPPORTED_EXTS:
            st.sidebar.warning(f"❌ Unsupported file: {filename}. Skipping.")
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(file_path):
            st.sidebar.info(f"📄 {filename} already exists. Skipping.")
            continue

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_file(file_path, file_ext)
        if not text or not text.strip():
            st.sidebar.warning(f"⚠️ No text found in {filename}. Skipping.")
            continue

        chunks = chunk_text(text)
        if not chunks:
            st.sidebar.warning(f"⚠️ Unable to chunk {filename}. Skipping.")
            continue

        embeddings = model.encode(chunks)
        all_chunks.extend(embeddings)
        all_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])
        summaries[filename] = summarize_text(text)

    if all_chunks:
        all_chunks = np.array(all_chunks).astype("float32")
        if index is None:
            dimension = all_chunks.shape[1]
            index = faiss.IndexFlatL2(dimension)
        index.add(all_chunks)
        combined_metadata = existing_metadata + all_metadata
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(combined_metadata, f)

    return summaries

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, file))
            except Exception:
                pass
