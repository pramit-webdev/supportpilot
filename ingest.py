import os
import pickle
import streamlit as st
from sentence_transformers import SentenceTransformer
from utils import chunk_text, summarize_text
import faiss
import numpy as np
import pdfplumber
from PIL import Image
import pandas as pd
import docx

from doctr.models import ocr_predictor

# === Setup ===
DATA_DIR = "data"
DOCS_DIR = os.path.join(DATA_DIR, "docs")
INDEX_DIR = os.path.join(DATA_DIR, "faiss_index")
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

SUPPORTED_EXTS = [".pdf", ".docx", ".doc", ".csv", ".xlsx", ".png", ".jpg", ".jpeg"]

model = SentenceTransformer("all-mpnet-base-v2")

# Initialize docTR (load once)
ocr_model = ocr_predictor(pretrained=True)

# Ensure directories exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)


# === OCR FUNCTION ===
def doctr_ocr_image_to_text(pil_img):
    """
    Extract text from a PIL image using docTR OCR.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    result = ocr_model([img_rgb])
    words = []
    for block in result.pages[0].blocks:
        for line in block.lines:
            words.append(line.text)
    return "\n".join(words)


# === FILE TEXT EXTRACTION ===
def extract_text_from_file(file_path, file_ext):
    ext = file_ext.lower()
    text = ""

    try:
        # PDF (including scanned)
        if ext == ".pdf":
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                    else:
                        pil_img = page.to_image(resolution=300).original.convert("RGB")
                        ocr_result = doctr_ocr_image_to_text(pil_img)
                        text += ocr_result + "\n"

        # Word Documents
        elif ext in [".docx", ".doc"]:
            doc = docx.Document(file_path)
            text += "\n".join(para.text for para in doc.paragraphs)

        # CSV/Excel
        elif ext in [".csv", ".xlsx"]:
            try:
                if ext == ".csv":
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path)
                text = df.to_string(index=False)
            except Exception as e:
                print(f"Error reading table: {e}")

        # Images (.jpg, .png, .jpeg)
        elif ext in [".png", ".jpg", ".jpeg"]:
            pil_img = Image.open(file_path).convert("RGB")
            text = doctr_ocr_image_to_text(pil_img)

    except Exception as e:
        print(f"[ERROR] Failed to extract text from {file_path}: {e}")

    return text


# === UPLOAD & INDEXING FUNCTION ===
def handle_upload(uploaded_files):
    summaries = {}
    all_chunks, all_metadata = [], []

    # Load existing FAISS index
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing_metadata = pickle.load(f)
    else:
        index, existing_metadata = None, []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        file_ext = os.path.splitext(filename)[1].lower()

        if file_ext not in SUPPORTED_EXTS:
            st.sidebar.warning(f"❌ Unsupported file type: {filename}")
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(file_path):
            st.sidebar.info(f"📁 {filename} already processed. Skipping.")
            continue

        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Extract text
        extracted_text = extract_text_from_file(file_path, file_ext)
        if not extracted_text.strip():
            st.sidebar.warning(f"⚠️ No text found in {filename}. Skipping.")
            continue

        # Chunk & embed
        chunks = chunk_text(extracted_text)
        if not chunks:
            st.sidebar.warning(f"⚠️ Failed to chunk {filename}. Skipping.")
            continue

        embeddings = model.encode(chunks)
        all_chunks.extend(embeddings)
        all_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])

        # Summarize for UI
        summaries[filename] = summarize_text(extracted_text)

    # Build FAISS index
    if all_chunks:
        all_chunks = np.array(all_chunks).astype("float32")
        if index is None:
            index = faiss.IndexFlatL2(all_chunks.shape[1])
        index.add(all_chunks)

        combined_metadata = existing_metadata + all_metadata
        faiss.write_index(index, INDEX_FILE)
        with open(METADATA_FILE, "wb") as f:
            pickle.dump(combined_metadata, f)

    return summaries


# === RESET FUNCTION ===
def reset_index():
    for folder in [DOCS_DIR, INDEX_DIR]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                continue
