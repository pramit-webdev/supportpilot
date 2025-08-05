# ✅ voice_app.py – BharatPilot (Gradio-based Voice + Document Assistant)

import os
import tempfile
import pickle
import faiss
import asyncio
import gradio as gr
import numpy as np
import pandas as pd
import pdfplumber
import docx
import requests
from datetime import datetime
from deepgram import DeepgramClient, SpeakOptions, PrerecordedOptions, FileSource
from sentence_transformers import SentenceTransformer
from textwrap import shorten

# ENV VARS
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Paths
INDEX_FILE = "data/faiss_index/support_index.faiss"
METADATA_FILE = "data/faiss_index/metadata.pkl"
DOCS_DIR = "data/docs"

# Ensure dirs exist
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs("data/faiss_index", exist_ok=True)

# Embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Init Deepgram
dg_client = DeepgramClient(DEEPGRAM_API_KEY)

# Utility: Chunking
def chunk_text(text, max_length=1000, overlap=150):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

# Utility: Summarize
def summarize_text(text, width=350):
    summary = shorten(text.strip(), width=width, placeholder="...")
    return f"Summary: {summary}"

# Extractor
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

# Upload and index
metadata = []
def handle_upload(files):
    new_chunks = []
    new_metadata = []
    for file in files:
        filename = file.name
        ext = os.path.splitext(filename)[1].lower()
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "wb") as f:
            f.write(file.read())
        text = extract_text(path, ext)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        embeddings = model.encode(chunks)
        new_chunks.extend(embeddings)
        new_metadata.extend([{
            "source": filename,
            "text": chunk,
            "file_type": ext,
            "upload_date": str(datetime.now().date())
        } for chunk in chunks])
    if not new_chunks:
        return "No new text chunks indexed."
    xb = np.array(new_chunks).astype("float32")
    if os.path.exists(INDEX_FILE):
        index = faiss.read_index(INDEX_FILE)
        with open(METADATA_FILE, "rb") as f:
            existing = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(xb.shape[1])
        existing = []
    index.add(xb)
    all_metadata = existing + new_metadata
    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(all_metadata, f)
    return f"✅ Indexed {len(new_chunks)} chunks from {len(files)} files."

# RAG context retriever

def retrieve_context(query, k=10):
    if not os.path.exists(INDEX_FILE):
        return []
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, k)
    chunks = []
    for i in I[0]:
        if 0 <= i < len(metadata):
            entry = metadata[i]
            chunks.append(f"[{entry['source']}]: {entry['text']}")
    return chunks

# Call Groq LLM

def call_llm(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for document Q&A."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }
    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    return r.json()['choices'][0]['message']['content']

# Voice-to-text
async def transcribe(audio_path):
    with open(audio_path, 'rb') as f:
        source = FileSource(buffer=f, mimetype="audio/wav")
        options = PrerecordedOptions(punctuate=True)
        res = await dg_client.listen.prerecorded(source, options)
        return res['results']['channels'][0]['alternatives'][0]['transcript']

# Text-to-speech

def synthesize(text):
    speak_opts = SpeakOptions(model="aura-2-thalia-en", encoding="mp3")
    res = dg_client.speak.v("v1").speak({"text": text}, speak_opts)
    return res.content  # MP3 bytes

# Full pipeline

def process_voice(audio):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio)
        path = tmp.name
    transcript = asyncio.run(transcribe(path))
    context = retrieve_context(transcript)
    prompt = f"Context:\n{chr(10).join(context)}\n\nQuestion: {transcript}\nAnswer:"
    answer = call_llm(prompt)
    tts = synthesize(answer)
    return transcript, answer, (tts, "output.mp3")

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ BharatPilot – Voice & Document Assistant")
    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_types=[".pdf", ".docx", ".csv", ".xlsx"], file_count="multiple")
            upload_btn = gr.Button("Upload & Index")
            upload_output = gr.Textbox(label="Upload status")
        with gr.Column():
            mic = gr.Audio(source="microphone", type="filepath", label="Speak your question")
            run_btn = gr.Button("Ask via Voice")
            transcript = gr.Textbox(label="Transcription")
            answer = gr.Textbox(label="Answer")
            voice = gr.Audio(label="Spoken Answer")

    upload_btn.click(fn=handle_upload, inputs=[file_input], outputs=[upload_output])
    run_btn.click(fn=process_voice, inputs=[mic], outputs=[transcript, answer, voice])

if __name__ == "__main__":
    demo.launch()
