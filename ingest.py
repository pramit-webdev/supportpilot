import os
import faiss
import pickle
import pdfplumber
import docx
import pandas as pd
from sentence_transformers import SentenceTransformer
from utils import chunk_text, summarize_text
import streamlit as st

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

SUPPORTED_EXTS = [".pdf", ".docx", ".doc", ".csv", ".xlsx"]

model = SentenceTransformer("all-mpnet-base-v2")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text(file_path, ext):
    try:
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif ext in [".docx", ".doc"]:
            doc_file = docx.Document(file_path)
            return "\n".join(p.text for p in doc_file.paragraphs)
        elif ext == ".csv":
            return pd.read_csv(file_path).to_markdown(index=False)
        elif ext == ".xlsx":
            return pd.read_excel(file_path).to_markdown(index=False)
    except Exception as e:
        return f"❌ Error reading {os.path.basename(file_path)}: {e}"

def handle_upload(uploaded_files):
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    summaries = {}
    all_chunks = []
    new_metadata = []
    for file in uploaded_files:
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTS:
            st.sidebar.warning(f"❌ Unsupported file type: {filename}")
            continue
        file_path = os.path.join(DOCS_DIR, filename)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        text = extract_text(file_path, ext)
        if not text or not text.strip():
            st.warning(f"⚠️ No text extracted from {filename}. Skipping.")
            continue
        chunks = chunk_text(text)
        if not chunks:
            st.warning(f"⚠️ Unable to split text from {filename}. Skipping.")
            continue
        embeddings = model.encode(chunks)
        all_chunks.extend(embeddings)
        new_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])
        summaries[filename] = summarize_text(text)
        st.success(f"✅ {filename} indexed.")
    if not all_chunks:
        return summaries
    all_chunks = np.array(all_chunks).astype("float32")
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        dimension = all_chunks.shape[1]
        index = faiss.IndexFlatL2(dimension)
        existing_metadata = []
    index.add(all_chunks)
    all_metadata = existing_metadata + new_metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_metadata, f)
    return summaries

def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))
    st.session_state.files_indexed = False
    st.session_state.messages = []
