from textwrap import shorten

def chunk_text(text, max_length=750, overlap=100):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

def summarize_text(text, width=300):
    summary = shorten(text.strip(), width=width, placeholder="...")
    return f"Summary: {summary}"
