# utils.py

def chunk_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks
