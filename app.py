# ✅ app.py (voice + document assistant using Deepgram STT + TTS)

import os
import streamlit as st
from ingest import handle_upload, reset_index
from rag import get_answer
from deepgram import Deepgram
import tempfile
import asyncio
import requests

# Streamlit config
st.set_page_config(page_title="BharatPilot", layout="wide")
st.title("🎙️ BharatPilot – Voice & Document AI Assistant")

# Constants
INDEX_PATH = "data/faiss_index/support_index.faiss"
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")

# Initialize Deepgram SDK
dg_client = Deepgram(DG_API_KEY) if DG_API_KEY else None

# Audio → Text
def transcribe_audio(audio_path):
    with open(audio_path, 'rb') as audio:
        source = {'buffer': audio, 'mimetype': 'audio/wav'}
        response = asyncio.run(dg_client.transcription.prerecorded(source, {'punctuate': True}))
        return response['results']['channels'][0]['alternatives'][0]['transcript']

# Text → Audio using Deepgram Aura TTS
def synthesize_speech_deepgram(text, model="aura-2-thalia-en"):
    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DG_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model": model
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.content  # MP3 bytes

# Session state
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# Sidebar: Upload & Filters
with st.sidebar:
    st.header("📁 Upload Your Files")
    uploaded_files = st.file_uploader(
        "Supported: PDF, DOCX, CSV, XLSX",
        type=["pdf", "docx", "doc", "csv", "xlsx"],
        accept_multiple_files=True
    )

    if st.button("🗑️ Reset All Documents"):
        reset_index()
        st.session_state.indexed_files = set()
        st.warning("All files and indexes have been cleared.")
        st.rerun()

    if st.session_state.indexed_files:
        st.sidebar.markdown("### 🔍 Filter Search Scope")
        selected_files = st.sidebar.multiselect(
            "Search only in these files:",
            options=list(st.session_state.indexed_files),
            default=list(st.session_state.indexed_files)
        )
        selected_types = st.sidebar.multiselect(
            "File types:",
            options=[".pdf", ".docx", ".csv", ".xlsx"],
            default=[]
        )
    else:
        selected_files = []
        selected_types = []

# Index new uploads
new_files = []
if uploaded_files:
    for file in uploaded_files:
        if file.name not in st.session_state.indexed_files:
            new_files.append(file)

    if new_files:
        with st.spinner("Indexing new files..."):
            summaries = handle_upload(new_files)
        for file in new_files:
            st.session_state.indexed_files.add(file.name)
        for fname, summary in summaries.items():
            st.sidebar.success(f"Indexed: {fname}")
            with st.sidebar.expander(f"Summary - {fname}"):
                st.markdown(summary)

if not os.path.exists(INDEX_PATH):
    st.info("Upload documents to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "ai", "content": "Hi! Upload files and ask a question about their contents."}
    ]

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Text chat input
if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("ai"):
        with st.spinner("Finding answer..."):
            answer, context = get_answer(user_input, selected_files, selected_types)
            st.markdown(answer)
            with st.expander("📄 Context Chunks"):
                for i, chunk in enumerate(context, 1):
                    st.markdown(f"**Chunk {i}**\n\n{chunk}")
    st.session_state.messages.append({"role": "ai", "content": answer})

# Voice chat input
st.markdown("---")
st.subheader("🎙️ Voice Chat (Ask using voice)")

audio_input = st.file_uploader("Upload a voice query (.wav)", type=["wav"])

if audio_input and DG_API_KEY:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_input.read())
        audio_path = tmp.name

    with st.spinner("Transcribing..."):
        try:
            transcript = transcribe_audio(audio_path)
            st.markdown(f"**You said:** {transcript}")
        except Exception as e:
            st.error(f"❌ Deepgram STT failed: {e}")
            st.stop()

    with st.spinner("Answering..."):
        answer, context = get_answer(transcript, selected_files, selected_types)
        st.markdown(f"**Answer:** {answer}")
        with st.expander("📄 Context Chunks"):
            for i, chunk in enumerate(context, 1):
                st.markdown(f"**Chunk {i}**\n\n{chunk}")

    with st.spinner("Synthesizing voice..."):
        try:
            audio_bytes = synthesize_speech_deepgram(answer)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"❌ Deepgram TTS failed: {e}")
