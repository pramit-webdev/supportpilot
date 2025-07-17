import os
import pickle
import faiss
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
import pytesseract
from PIL import Image
from utils import chunk_text, summarize_text
import docx
import pandas as pd

DOCS_DIR = "data/docs"
INDEX_DIR = "data/faiss_index"
INDEX_FILE = os.path.join(INDEX_DIR, "support_index.faiss")
METADATA_FILE = os.path.join(INDEX_DIR, "metadata.pkl")

SUPPORTED_EXTS = [".pdf", ".docx", ".doc", ".csv", ".png", ".jpg", ".jpeg"]

model = SentenceTransformer("all-MiniLM-L6-v2")

os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)

def extract_text_from_file(file_path, file_ext):
    text = ""

    if file_ext.lower() == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for page_number, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                else:
                    try:
                        pil_image = page.to_image(resolution=300).original.convert("RGB")
                        ocr_result = pytesseract.image_to_string(pil_image)
                        text += ocr_result + "\n"
                    except Exception as e:
                        print(f"OCR error on PDF page {page_number}: {e}")

    elif file_ext.lower() in [".docx", ".doc"]:
        try:
            doc = docx.Document(file_path)
            for para in doc.paragraphs:
                text += para.text + "\n"
        except Exception as e:
            print(f"Error processing Word file: {e}")

    elif file_ext.lower() == ".csv":
        try:
            df = pd.read_csv(file_path)
            text = df.to_string(index=False)
        except Exception as e:
            print(f"Error processing CSV file: {e}")

    elif file_ext.lower() in [".png", ".jpg", ".jpeg"]:
        try:
            pil_image = Image.open(file_path).convert("RGB")
            ocr_result = pytesseract.image_to_string(pil_image)
            text += ocr_result + "\n"
        except Exception as e:
            print(f"OCR error on image file: {e}")

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
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in SUPPORTED_EXTS:
            st.sidebar.warning(f"Unsupported file type: {filename}. Skipping.")
            continue

        file_path = os.path.join(DOCS_DIR, filename)
        if os.path.exists(file_path):
            st.sidebar.info(f"📌 {filename} already indexed. Skipping.")
            continue

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        text = extract_text_from_file(file_path, file_ext)
        if not text or not text.strip():
            st.sidebar.warning(f"⚠️ No text found in {filename}. Skipping.")
            continue

        chunks = chunk_text(text)
        embeddings = model.encode(chunks)
        all_chunks.extend(embeddings)
        all_metadata.extend([{"source": filename, "text": chunk} for chunk in chunks])
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
