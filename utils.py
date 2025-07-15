from textwrap import shorten

def chunk_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

def summarize_text(text):
    summary = shorten(text, width=300, placeholder="...")
    return f"Summary: {summary}"
