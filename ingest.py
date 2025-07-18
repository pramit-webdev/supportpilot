import os
import faiss
import pickle
import pdfplumber
import docx
import pandas as pd
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from utils import chunk_text
import streamlit as st

# Paths
DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Supported formats
SUPPORTED_IMAGE_TYPES = [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Ensure folders exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

# --- Extraction Functions ---

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"❌ Extraction error in {os.path.basename(file_path)}: {e}"

def extract_text_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"❌ Extraction error in {os.path.basename(file_path)}: {e}"

def extract_text_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_string(index=False)  # Avoid tabulate dependency
    except Exception as e:
        return f"❌ Extraction error in {os.path.basename(file_path)}: {e}"

def extract_text_from_image(file_path):
    try:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"❌ OCR extraction error in {os.path.basename(file_path)}: {e}"

# --- Handle Uploads ---

def handle_upload(uploaded_files):
    existing_files = set(os.listdir(DOCS_DIR))
    all_chunks = []
    new_metadata = []
    summaries = {}

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        if filename in existing_files:
            st.info(f"ℹ️ {filename} is already indexed. Skipping.")
            continue

        # Save file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Extract based on file type
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext == ".docx":
            text = extract_text_from_docx(file_path)
        elif ext == ".csv":
            text = extract_text_from_csv(file_path)
        elif ext in SUPPORTED_IMAGE_TYPES:
            text = extract_text_from_image(file_path)
        else:
            st.warning(f"⚠️ Unsupported file type: {filename}")
            continue

        if not text or text.strip() == "":
            st.warning(f"⚠️ No text extracted from {filename}. Skipping.")
            continue

        # Chunking
        chunks = chunk_text(text)
        if not chunks:
            st.warning(f"⚠️ No valid chunks generated from {filename}. Skipping.")
            continue

        all_chunks.extend(chunks)
        new_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        # Generate summary (basic first paragraph)
        summaries[filename] = text.strip().split("\n")[0][:500]
        st.success(f"✅ {filename} uploaded and indexed.")

    # Return if no valid chunks
    if not all_chunks:
        return summaries

    # Encode all new chunks
    new_embeddings = model.encode(all_chunks, batch_size=32, show_progress_bar=False)

    # Load existing index + metadata (if available)
    if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        dimension = new_embeddings[0].shape[0]
        index = faiss.IndexFlatL2(dimension)
        existing_metadata = []

    # Add new data to index
    index.add(new_embeddings)
    all_metadata = existing_metadata + new_metadata

    # Save index and metadata
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
