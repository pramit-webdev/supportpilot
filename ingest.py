import os, pickle, streamlit as st, faiss, numpy as np, pdfplumber, pandas as pd, docx
from sentence_transformers import SentenceTransformer
from utils import chunk_text, summarize_text

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
                    # Add fallback: PDF text extraction fails often on tables!
                    if not page_text or not page_text.strip():
                        st.sidebar.warning(f"⚠️ PDF extraction failed on some pages in {os.path.basename(file_path)}. Consider exporting as .docx or .csv.")
                    else:
                        text += page_text + "\n"
        elif ext in [".docx", ".doc"]:
            docfile = docx.Document(file_path)
            text += "\n".join(par.text for par in docfile.paragraphs)
        elif ext == ".csv":
            # Convert CSVs to Markdown tables for better semantic context
            df = pd.read_csv(file_path)
            text = df.to_markdown(index=False)
        elif ext == ".xlsx":
            df = pd.read_excel(file_path)
            text = df.to_markdown(index=False)
    except Exception as e:
        st.sidebar.error(f"❌ Extraction error in {os.path.basename(file_path)}: {e}")
        print(f"Text extraction error for {file_path}: {e}")
    print(f"🟢 Extracted text from {os.path.basename(file_path)}:\n{text[:500]}...\n")
    return text

def handle_upload(uploaded_files):
    summaries = {}
    all_chunks, all_metadata = [], []
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
        print(f"🧩 Chunks for {filename} (showing up to 2):")
        for idx, chunk in enumerate(chunks[:2]):
            print(f"-- Chunk {idx+1}: {chunk[:250]}...\n")
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
