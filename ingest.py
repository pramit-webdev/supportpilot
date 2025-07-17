import os
import pickle
import faiss
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from utils import chunk_text, summarize_text
import pytesseract
from PIL import Image
import io


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
        for page_number, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text + "\n"
            else:
                # OCR fallback for scanned/image-only pages
                try:
                    pil_image = page.to_image(resolution=300).original.convert("RGB")
                    ocr_result = pytesseract.image_to_string(pil_image)
                    if ocr_result.strip():
                        text += ocr_result + "\n"
                except Exception as e:
                    print(f"OCR error on page {page_number}: {e}")
    return text

def handle_upload(uploaded_files):
    summaries = {}
    all_chunks = []
    all_metadata = []

    # Load existing index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        index = None
        existing_metadata = []

    for file in uploaded_files:
        filename = file.name
        file_path = os.path.join(DOCS_DIR, filename)

        if os.path.exists(file_path):
            st.sidebar.info(f"📌 {filename} already indexed. Skipping.")
            continue

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            st.sidebar.warning(f"⚠️ No text found in {filename}. Skipping.")
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks)

        # Append new data
        all_chunks.extend(embeddings)
        all_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        # Generate summary
        summary = summarize_text(text)
        summaries[filename] = summary

    if all_chunks:
        import numpy as np
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
            os.remove(os.path.join(folder, file))
