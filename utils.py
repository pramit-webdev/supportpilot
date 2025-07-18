from textwrap import shorten

def chunk_text(text, max_length=1000, overlap=150):
    # Increased chunk size and overlap for full context retention—excellent for multi-sentence/tabular/cell answers
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

def summarize_text(text, width=350):
    summary = shorten(text.strip(), width=width, placeholder="...")
    return f"Summary: {summary}"
