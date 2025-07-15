from transformers import pipeline

def chunk_text(text, max_length=500, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_length]
        chunks.append(" ".join(chunk))
        i += max_length - overlap
    return chunks

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def generate_summary(text):
    if not text.strip():
        return "No summary available."
    try:
        summary = summarizer(text[:3000], max_length=100, min_length=30, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        return "Summary generation failed."
