import os
import faiss
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from utils import chunk_text
from pathlib import Path

# Paths
DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create folders if missing
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def handle_upload(uploaded_files):
    all_chunks = []
    metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        # Save uploaded file
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Extract and chunk text
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)

        all_chunks.extend(chunks)
        metadata.extend([
            {"source": filename, "text": chunk} for chunk in chunks
        ])

    # Create embeddings
    embeddings = model.encode(all_chunks)

    # Build FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save index and metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

def reset_index():
    # Clear FAISS index and docs
    for folder in [DOCS_DIR, INDEX_DIR]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
